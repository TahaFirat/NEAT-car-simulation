import math
import random
import sys
import os

import neat # type: ignore
import pygame

# --- CONSTANTS ---
WINDOW_WIDTH = 1600  # Increased
WINDOW_HEIGHT = 900 # Increased
VIZ_PANEL_WIDTH = 340; VIZ_PANEL_X_OFFSET = 10; VIZ_PANEL_Y_OFFSET = 10
VIZ_PANEL_HEIGHT = WINDOW_HEIGHT - (2 * VIZ_PANEL_Y_OFFSET)
STATS_PANEL_WIDTH = 200; STATS_PANEL_X_OFFSET = VIZ_PANEL_X_OFFSET + VIZ_PANEL_WIDTH + 10
GAME_AREA_X_OFFSET = STATS_PANEL_X_OFFSET + STATS_PANEL_WIDTH + 10
# Width for the right panel (buttons, max speed text)
RIGHT_PANEL_AREA_WIDTH = 160
GAME_AREA_WIDTH = WINDOW_WIDTH - GAME_AREA_X_OFFSET - RIGHT_PANEL_AREA_WIDTH
GAME_AREA_HEIGHT = int(WINDOW_HEIGHT * 0.9); GAME_AREA_Y_OFFSET = (WINDOW_HEIGHT - GAME_AREA_HEIGHT) // 2
CAR_SCALE_FACTOR = 0.8; CAR_SIZE_X = int(40 * CAR_SCALE_FACTOR); CAR_SIZE_Y = int(40 * CAR_SCALE_FACTOR)
CAR_IMAGE_PATH = 'car1.png'; ORIGINAL_GAME_WIDTH_FOR_POS = 1280; ORIGINAL_GAME_HEIGHT_FOR_POS = 720
INITIAL_POS_X_ORIGINAL = 660; INITIAL_POS_Y_ORIGINAL = 610
INITIAL_POSITION = [
    int(INITIAL_POS_X_ORIGINAL * (GAME_AREA_WIDTH / ORIGINAL_GAME_WIDTH_FOR_POS)),
    int(INITIAL_POS_Y_ORIGINAL * (GAME_AREA_HEIGHT / ORIGINAL_GAME_HEIGHT_FOR_POS))
]
ANGLE_STEP = 10; ACTION_LABELS = ["Left", "Right", "Decelerate", "Accelerate"] # Translated
RADAR_ANGLES = [-90, -45, 0, 45, 90]
MAX_RADAR_DISTANCE = int(200 * (GAME_AREA_WIDTH / ORIGINAL_GAME_WIDTH_FOR_POS) * 0.8)
RADAR_NORMALIZATION_FACTOR = MAX_RADAR_DISTANCE

# --- COLOR CONSTANTS ---
OBSTACLE_WHITE_MIN_RGB = (240, 240, 240)
OBSTACLE_WHITE_MAX_RGB = (255, 255, 255)

FINISH_LINE_GREEN_COLOR_RGBA = (56, 111, 56, 255)
ROAD_COLOR_RGBA = (0, 0, 0, 255)
CENTER_LINE_COLOR_RGBA = (100, 100, 100, 255)

BUTTON_COLOR = (40,45,70); BUTTON_HOVER_COLOR = (60,70,90); BUTTON_TEXT_COLOR = (200,220,255)
RADAR_VIS_COLOR = (0,200,200); TEXT_COLOR = (200,200,220)
INFO_TEXT_BACKGROUND_COLOR = (25,30,40,200); SELECTED_ACTION_COLOR = (0,255,150)
FPS = 60; GENERATION_TIME_LIMIT_SECONDS = 70; CONFIG_PATH = "./config.txt"

global_initial_speed = 5.0; global_min_speed = 2.0; global_max_speed = 10.0; global_speed_step = 1.5
user_quit_simulation = False
user_requested_speed_change = 0.0 # For button-based speed adjustments

STAGNATION_CHECK_INTERVAL = FPS // 2
STAGNATION_THRESHOLD_DISTANCE = CAR_SIZE_X * 0.10
STAGNATION_FRAMES_LIMIT = FPS * 25

DISTANCE_FITNESS_MULTIPLIER = 2.0; TIME_FITNESS_MULTIPLIER = 0.01
LANE_REWARD_FACTOR = 0.025; CENTER_LINE_PENALTY_FACTOR = 0.018
WRONG_LANE_PENALTY_FACTOR = 0.06

NN_PANEL_BG_COLOR = (15,20,30,235); NODE_RADIUS = 16
NODE_BASE_FILL_COLOR = (40,50,65); NODE_BORDER_THICKNESS = 2; NODE_BORDER_COLOR = (70,80,100)
NODE_ACTIVE_POSITIVE_FILL = (0,220,220); NODE_ACTIVE_NEGATIVE_FILL = (255,50,100)
NODE_DOTTED_OUTLINE_COLOR = (0,180,180); CONNECTION_POSITIVE_COLOR = (0,200,200)
CONNECTION_NEGATIVE_COLOR = (230,40,90); MAX_WEIGHT_FOR_THICKNESS = 2.5
CONNECTION_MAX_THICKNESS = 5; CONNECTION_DASH_LEN = 5; CONNECTION_GAP_LEN = 3
ARROW_COLOR = (220,220,240); ARROW_SIZE_FACTOR = 0.65
ARROW_POINTS_UP_NORM = [(0,-0.5),(-0.35,0.25),(0,0.05),(0.35,0.25)]; ACTIVATION_THRESHOLD = 0.1
DEBUG_DRAW_CAR_CORNERS = True

def is_color_in_range(pixel_rgba, min_rgb, max_rgb):
    r, g, b, _ = pixel_rgba
    return (min_rgb[0] <= r <= max_rgb[0] and
            min_rgb[1] <= g <= max_rgb[1] and
            min_rgb[2] <= b <= max_rgb[2])

def rotate_points(points, angle_degrees, origin=(0,0)):
    angle_rad = math.radians(angle_degrees); cos_a = math.cos(angle_rad); sin_a = math.sin(angle_rad)
    rotated_points = []
    for p_x, p_y in points:
        x = p_x - origin[0]; y = p_y - origin[1]
        new_x = x * cos_a - y * sin_a + origin[0]
        new_y = x * sin_a + y * cos_a + origin[1]
        rotated_points.append((new_x, new_y))
    return rotated_points

def draw_node_arrow(surface, center_x, center_y, radius, direction_angle_deg, color):
    scaled_radius = radius * ARROW_SIZE_FACTOR
    base_arrow_points = [(p_x * scaled_radius, p_y * scaled_radius) for p_x, p_y in ARROW_POINTS_UP_NORM]
    rotated_arrow_points = rotate_points(base_arrow_points, direction_angle_deg)
    final_arrow_points = [(p_x + center_x, p_y + center_y) for p_x, p_y in rotated_arrow_points]
    if final_arrow_points:
        pygame.draw.polygon(surface, color, final_arrow_points)
        pygame.draw.lines(surface, (color[0]//2, color[1]//2, color[2]//2), True, final_arrow_points, 1)

def draw_neat_visualization(screen, genome, config, in_vals, out_vals, viz_rect):
    viz_surface = pygame.Surface(viz_rect.size, pygame.SRCALPHA)
    viz_surface.fill(NN_PANEL_BG_COLOR)

    if genome is None or config is None:
        screen.blit(viz_surface, viz_rect.topleft)
        return

    input_keys = list(config.genome_config.input_keys)
    output_keys = list(config.genome_config.output_keys)

    current_input_values = list(in_vals) if in_vals else [0.0] * len(input_keys)
    current_output_values = list(out_vals) if out_vals else [0.0] * len(output_keys)

    while len(current_input_values) < len(input_keys): current_input_values.append(0.0)
    while len(current_output_values) < len(output_keys): current_output_values.append(0.0)

    node_positions, node_render_properties = {}, {}
    margin_x, margin_y = viz_rect.width * 0.1, viz_rect.height * 0.05
    drawable_width, drawable_height = viz_rect.width - 2 * margin_x, viz_rect.height - 2 * margin_y

    num_input_nodes = len(input_keys)
    y_input_nodes = margin_y + NODE_RADIUS
    for i, key_input in enumerate(input_keys):
        x_pos = margin_x + (drawable_width * (i + 0.5) / num_input_nodes if num_input_nodes > 0 else drawable_width * 0.5)
        node_positions[key_input] = (int(x_pos), int(y_input_nodes))
        activation = current_input_values[i] if i < len(current_input_values) else 0.0
        fill_color = NODE_ACTIVE_POSITIVE_FILL if activation > ACTIVATION_THRESHOLD else NODE_BASE_FILL_COLOR
        node_render_properties[key_input] = {'fill': fill_color, 'dotted': False}

    num_output_nodes = len(output_keys)
    y_output_nodes = viz_rect.height - margin_y - NODE_RADIUS
    for i, key_output in enumerate(output_keys):
        x_pos = margin_x + (drawable_width * (i + 0.5) / num_output_nodes if num_output_nodes > 0 else drawable_width * 0.5)
        node_positions[key_output] = (int(x_pos), int(y_output_nodes))
        activation = current_output_values[i] if i < len(current_output_values) else 0.0
        fill_color = NODE_BASE_FILL_COLOR
        if activation > ACTIVATION_THRESHOLD: fill_color = NODE_ACTIVE_POSITIVE_FILL
        elif activation < -ACTIVATION_THRESHOLD: fill_color = NODE_ACTIVE_NEGATIVE_FILL
        node_render_properties[key_output] = {'fill': fill_color, 'dotted': False}

    hidden_keys = sorted([k for k in genome.nodes if k not in input_keys and k not in output_keys])
    num_hidden_nodes = len(hidden_keys)
    if num_hidden_nodes > 0:
        max_hidden_rows_per_col = 6
        num_hidden_cols = (num_hidden_nodes + max_hidden_rows_per_col - 1) // max_hidden_rows_per_col
        hidden_area_total_height = drawable_height * 0.6
        space_between_io_layers = y_output_nodes - (y_input_nodes + 2 * NODE_RADIUS)
        if space_between_io_layers < hidden_area_total_height:
            hidden_area_total_height = max(space_between_io_layers * 0.8, NODE_RADIUS * 2 * max_hidden_rows_per_col)
        y_hidden_block_start = y_input_nodes + NODE_RADIUS + (space_between_io_layers - hidden_area_total_height) / 2
        for i, key_hidden in enumerate(hidden_keys):
            col_index, index_in_col = i // max_hidden_rows_per_col, i % max_hidden_rows_per_col
            nodes_in_current_col = min(max_hidden_rows_per_col, num_hidden_nodes - col_index * max_hidden_rows_per_col)
            if nodes_in_current_col == 0: nodes_in_current_col = max_hidden_rows_per_col
            x_pos_hidden = margin_x + (drawable_width * (col_index + 0.5) / num_hidden_cols if num_hidden_cols > 0 else drawable_width * 0.5)
            y_pos_hidden = y_hidden_block_start + (hidden_area_total_height * (index_in_col + 0.5) / nodes_in_current_col if nodes_in_current_col > 0 else hidden_area_total_height * 0.5)
            node_positions[key_hidden] = (int(x_pos_hidden), int(y_pos_hidden))
            bias_val = genome.nodes[key_hidden].bias
            normalized_bias_effect = math.tanh(bias_val / 2.0)
            fill_color, use_dotted_outline = NODE_BASE_FILL_COLOR, True
            if normalized_bias_effect > ACTIVATION_THRESHOLD: fill_color, use_dotted_outline = NODE_ACTIVE_POSITIVE_FILL, False
            elif normalized_bias_effect < -ACTIVATION_THRESHOLD: fill_color, use_dotted_outline = NODE_ACTIVE_NEGATIVE_FILL, False
            node_render_properties[key_hidden] = {'fill': fill_color, 'dotted': use_dotted_outline}

    for conn_gene in genome.connections.values():
        if not conn_gene.enabled: continue
        input_node_key, output_node_key = conn_gene.key
        if input_node_key not in node_positions or output_node_key not in node_positions: continue
        point_input, point_output = node_positions[input_node_key], node_positions[output_node_key]
        weight = conn_gene.weight
        line_color = CONNECTION_POSITIVE_COLOR if weight > 0 else CONNECTION_NEGATIVE_COLOR
        thickness_ratio = min(1.0, abs(weight) / MAX_WEIGHT_FOR_THICKNESS)
        line_thickness = max(1, int(1 + thickness_ratio * (CONNECTION_MAX_THICKNESS - 1)))
        dx, dy = point_output[0] - point_input[0], point_output[1] - point_input[1]
        distance = math.hypot(dx, dy)
        if distance == 0: continue
        num_dashes = int(distance / (CONNECTION_DASH_LEN + CONNECTION_GAP_LEN))
        if num_dashes <= 0: pygame.draw.line(viz_surface, line_color, point_input, point_output, line_thickness)
        else:
            for i_dash in range(num_dashes):
                start_ratio = (i_dash * (CONNECTION_DASH_LEN + CONNECTION_GAP_LEN)) / distance
                end_ratio = min((i_dash * (CONNECTION_DASH_LEN + CONNECTION_GAP_LEN) + CONNECTION_DASH_LEN) / distance, 1.0)
                start_point_dash = (point_input[0] + dx * start_ratio, point_input[1] + dy * start_ratio)
                end_point_dash = (point_input[0] + dx * end_ratio, point_input[1] + dy * end_ratio)
                pygame.draw.line(viz_surface, line_color, start_point_dash, end_point_dash, line_thickness)

    for node_key, node_pos in node_positions.items():
        properties = node_render_properties.get(node_key, {'fill': NODE_BASE_FILL_COLOR, 'dotted': True})
        pygame.draw.circle(viz_surface, properties['fill'], node_pos, NODE_RADIUS)
        pygame.draw.circle(viz_surface, NODE_BORDER_COLOR, node_pos, NODE_RADIUS, NODE_BORDER_THICKNESS)
        if properties['dotted']:
            for i_dot in range(10):
                angle_dot = 2 * math.pi * i_dot / 10
                dot_x = int(node_pos[0] + (NODE_RADIUS - NODE_BORDER_THICKNESS/2) * math.cos(angle_dot))
                dot_y = int(node_pos[1] + (NODE_RADIUS - NODE_BORDER_THICKNESS/2) * math.sin(angle_dot))
                pygame.draw.circle(viz_surface, NODE_DOTTED_OUTLINE_COLOR, (dot_x, dot_y), 1)
        arrow_direction_degrees = None
        if node_key in input_keys: arrow_direction_degrees = 180
        elif node_key in output_keys:
            try:
                node_index = output_keys.index(node_key)
                if node_index == 0: arrow_direction_degrees = 270
                elif node_index == 1: arrow_direction_degrees = 90
                elif node_index == 2: arrow_direction_degrees = 180
                elif node_index == 3: arrow_direction_degrees = 0
            except ValueError: pass
        if arrow_direction_degrees is not None:
            draw_node_arrow(viz_surface, node_pos[0], node_pos[1], NODE_RADIUS - NODE_BORDER_THICKNESS, arrow_direction_degrees, ARROW_COLOR)
    screen.blit(viz_surface, viz_rect.topleft)

class UserQuitException(Exception):pass
class Car:
    def __init__(self,car_sfc):
        self.sprite_original=car_sfc
        self.rotated_sprite=self.sprite_original
        self.position=list(INITIAL_POSITION)
        self.angle = 0.0
        self.target_angle = self.angle
        self.speed=global_initial_speed
        self.center=[self.position[0]+CAR_SIZE_X/2,self.position[1]+CAR_SIZE_Y/2]
        self.corners,self.radars,self.last_radar_data,self.last_nn_output=[],[],[],[]
        self.alive=True
        self.distance_driven,self.time_survived,self.stagnation_timer=0.0,0,0
        self.last_pos_for_stagnation_check=list(self.position)
        self.frames_since_last_stagnation_check=0
        self.angle_smoothing_factor=0.07
        self.target_speed = global_initial_speed
        self.speed_smoothing_factor = 0.05
        self.id = id(self)
        self.genome_key = "N/A_init"
        self.frames_in_correct_lane = 0
        self.frames_on_center_line = 0
        self.frames_in_wrong_lane_or_wall = 0

    def get_rect_on_screen(self):
        car_r_ig=pygame.Rect(self.position[0],self.position[1],CAR_SIZE_X,CAR_SIZE_Y)
        return car_r_ig.move(GAME_AREA_X_OFFSET,GAME_AREA_Y_OFFSET)

    def draw(self,screen_surface):
        drw_px=self.position[0]+GAME_AREA_X_OFFSET;drw_py=self.position[1]+GAME_AREA_Y_OFFSET
        sprite_rect = self.rotated_sprite.get_rect(center=(drw_px + CAR_SIZE_X / 2, drw_py + CAR_SIZE_Y / 2))
        screen_surface.blit(self.rotated_sprite, sprite_rect.topleft)
        if self.alive:self._draw_radars(screen_surface)
        if DEBUG_DRAW_CAR_CORNERS and self.alive:
            for corner_x, corner_y in self.corners:
                screen_corner_x = int(corner_x + GAME_AREA_X_OFFSET)
                screen_corner_y = int(corner_y + GAME_AREA_Y_OFFSET)
                pygame.draw.circle(screen_surface, (255, 255, 0, 180), (screen_corner_x, screen_corner_y), 3)

    def _draw_radars(self,screen_surface):
        cx_os,cy_os=self.center[0]+GAME_AREA_X_OFFSET,self.center[1]+GAME_AREA_Y_OFFSET
        for r_end_p,_ in self.radars:
            rex_os,rey_os=r_end_p[0]+GAME_AREA_X_OFFSET,r_end_p[1]+GAME_AREA_Y_OFFSET
            pygame.draw.line(screen_surface,RADAR_VIS_COLOR,(int(cx_os),int(cy_os)),(int(rex_os),int(rey_os)),2)
            pygame.draw.circle(screen_surface,RADAR_VIS_COLOR,(int(rex_os),int(rey_os)),4)

    def _check_collision(self,g_map_sfc):
        if not self.alive:return
        for corner_idx, (px,py) in enumerate([(int(p[0]),int(p[1])) for p in self.corners]):
            if not(0<=px<g_map_sfc.get_width() and 0<=py<g_map_sfc.get_height()):
                self.alive=False;return
            try:
                pixel_color_at_corner = g_map_sfc.get_at((px,py))
                if pixel_color_at_corner == FINISH_LINE_GREEN_COLOR_RGBA: pass
                elif is_color_in_range(pixel_color_at_corner, OBSTACLE_WHITE_MIN_RGB, OBSTACLE_WHITE_MAX_RGB):
                    self.alive=False;return
            except IndexError: self.alive=False;return

    def _update_radars(self,g_map_sfc):
        self.radars.clear();self.last_radar_data.clear()
        for deg_off in RADAR_ANGLES:
            radar_world_angle_math = self.angle + deg_off
            r_ang_r = math.radians(360 - radar_world_angle_math)
            l=0.0
            while l<MAX_RADAR_DISTANCE:
                x=int(self.center[0]+math.cos(r_ang_r)*l)
                y=int(self.center[1]+math.sin(r_ang_r)*l)
                if not(0<=x<g_map_sfc.get_width() and 0<=y<g_map_sfc.get_height()):break
                try:
                    pixel_color_radar = g_map_sfc.get_at((x,y))
                    if pixel_color_radar == FINISH_LINE_GREEN_COLOR_RGBA: pass
                    elif is_color_in_range(pixel_color_radar, OBSTACLE_WHITE_MIN_RGB, OBSTACLE_WHITE_MAX_RGB): break
                except IndexError:break
                l+=1.0
            fx,fy=int(self.center[0]+math.cos(r_ang_r)*l),int(self.center[1]+math.sin(r_ang_r)*l)
            self.radars.append([(fx,fy),l]);self.last_radar_data.append(l/RADAR_NORMALIZATION_FACTOR)

    def _check_stagnation(self):
        self.frames_since_last_stagnation_check+=1
        if self.frames_since_last_stagnation_check>=STAGNATION_CHECK_INTERVAL:
            moved_dist_since_last_check=math.hypot(self.position[0]-self.last_pos_for_stagnation_check[0],
                                                   self.position[1]-self.last_pos_for_stagnation_check[1])
            if moved_dist_since_last_check<STAGNATION_THRESHOLD_DISTANCE: self.stagnation_timer+=self.frames_since_last_stagnation_check
            else: self.stagnation_timer=0
            self.last_pos_for_stagnation_check=list(self.position)
            self.frames_since_last_stagnation_check=0
        if self.stagnation_timer>=STAGNATION_FRAMES_LIMIT: self.alive=False
            
    def _check_lane_position(self, g_map_sfc):
        if not self.alive: return

        offset_dist = CAR_SIZE_X * 0.38 
        
        # Aracın hareket yönü (0 derece = Doğu/East, artan açılar Saat Yönünün Tersi/CCW)
        # Bu, self.angle'ın hareket için nasıl kullanıldığıyla tutarlı olmalı: ang_r_move = math.radians(360 - self.angle)
        math_heading_rad = math.radians(360 - self.angle)

        # Hareket yönüne göre "sağdaki" vektör: (-sin(theta), cos(theta))
        # sin(theta) x bileşeni olur, cos(theta) y bileşeni (standart matematikte)
        # Pygame ekran koordinatlarında Y aşağı doğru olduğu için Y bileşeni için işaret değişikliği gerekmez.
        sample_offset_x = -math.sin(math_heading_rad) * offset_dist
        sample_offset_y = math.cos(math_heading_rad) * offset_dist
        
        sample_x = int(self.center[0] + sample_offset_x)
        sample_y = int(self.center[1] + sample_offset_y)

        if 0 <= sample_x < g_map_sfc.get_width() and 0 <= sample_y < g_map_sfc.get_height():
            try:
                pixel_color_sample = g_map_sfc.get_at((sample_x, sample_y))
                if pixel_color_sample[0:3] == ROAD_COLOR_RGBA[0:3]: self.frames_in_correct_lane += 1
                elif pixel_color_sample[0:3] == CENTER_LINE_COLOR_RGBA[0:3]: self.frames_on_center_line += 1
                elif is_color_in_range(pixel_color_sample, OBSTACLE_WHITE_MIN_RGB, OBSTACLE_WHITE_MAX_RGB):
                    self.frames_in_wrong_lane_or_wall += 1
            except IndexError: self.frames_in_wrong_lane_or_wall += 1
        else: self.frames_in_wrong_lane_or_wall += 1

    def update(self,g_map_sfc):
        if not self.alive:return
        self._check_stagnation()
        if not self.alive: return

        ang_diff=(self.target_angle-self.angle+180)%360-180
        max_angle_change_this_frame = ANGLE_STEP * 0.45
        requested_angle_change = ang_diff * self.angle_smoothing_factor
        actual_angle_change = max(-max_angle_change_this_frame, min(requested_angle_change, max_angle_change_this_frame))
        self.angle = (self.angle + actual_angle_change) % 360

        speed_diff = self.target_speed - self.speed
        self.speed += speed_diff * self.speed_smoothing_factor
        self.speed = max(global_min_speed, min(self.speed, global_max_speed))

        self.rotated_sprite=self._rotate_center(self.sprite_original,self.angle)
        ang_r_move=math.radians(360-self.angle)
        self.position[0]+=math.cos(ang_r_move)*self.speed
        self.position[1]+=math.sin(ang_r_move)*self.speed
        self.position[0]=max(0,min(self.position[0], g_map_sfc.get_width()-CAR_SIZE_X))
        self.position[1]=max(0,min(self.position[1], g_map_sfc.get_height()-CAR_SIZE_Y))

        self.distance_driven+=abs(self.speed);self.time_survived+=1
        self.center=[self.position[0]+CAR_SIZE_X/2,self.position[1]+CAR_SIZE_Y/2]

        half_x, half_y = CAR_SIZE_X / 2, CAR_SIZE_Y / 2
        corners_relative_to_sprite_center = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
        self.corners.clear()
        angle_rad_for_point_rotation = math.radians(self.angle)
        cos_a, sin_a = math.cos(angle_rad_for_point_rotation), math.sin(angle_rad_for_point_rotation)
        for rel_x, rel_y in corners_relative_to_sprite_center:
            rotated_rel_x = rel_x * cos_a - rel_y * sin_a
            rotated_rel_y = rel_x * sin_a + rel_y * cos_a
            self.corners.append((self.center[0] + rotated_rel_x, self.center[1] + rotated_rel_y))

        self._check_collision(g_map_sfc)
        if not self.alive: return
        self._update_radars(g_map_sfc)
        self._check_lane_position(g_map_sfc)

    def get_data_for_nn(self):return self.last_radar_data if self.last_radar_data and len(self.last_radar_data)==len(RADAR_ANGLES) else [1.0]*len(RADAR_ANGLES)
    def is_alive(self):return self.alive

    def get_fitness(self):
        base_fit = self.distance_driven * DISTANCE_FITNESS_MULTIPLIER + self.time_survived * TIME_FITNESS_MULTIPLIER
        lane_bonus = self.frames_in_correct_lane * LANE_REWARD_FACTOR
        lane_penalty = (self.frames_on_center_line * CENTER_LINE_PENALTY_FACTOR) + \
                       (self.frames_in_wrong_lane_or_wall * WRONG_LANE_PENALTY_FACTOR)
        modified_fitness = base_fit + lane_bonus - lane_penalty
        if self.distance_driven < CAR_SIZE_X * 4 or self.time_survived < FPS * 3.5:
            early_death_penalty_factor = 0.08
            if (self.frames_in_wrong_lane_or_wall * WRONG_LANE_PENALTY_FACTOR * 1.5) > \
               (self.frames_in_correct_lane * LANE_REWARD_FACTOR):
                early_death_penalty_factor = 0.005
            return max(0, modified_fitness * early_death_penalty_factor)
        return max(0, modified_fitness)

    def _rotate_center(self, image_to_rotate, angle):
        original_rect = image_to_rotate.get_rect()
        rotated_image_intermediate = pygame.transform.rotate(image_to_rotate, angle)
        rotated_image_rect = rotated_image_intermediate.get_rect(center=original_rect.center)
        final_rotated_image = pygame.Surface(original_rect.size, pygame.SRCALPHA)
        final_rotated_image.fill((0, 0, 0, 0))
        blit_x = (original_rect.width - rotated_image_rect.width) // 2
        blit_y = (original_rect.height - rotated_image_rect.height) // 2
        final_rotated_image.blit(rotated_image_intermediate, (blit_x, blit_y))
        return final_rotated_image

class Button:
    def __init__(self,x,y,w,h,txt,fnt,col,h_col,act=None):self.rect,self.text,self.font,self.col,self.hov_col,self.act,self.is_hov=pygame.Rect(x,y,w,h),txt,fnt,col,h_col,act,False
    def draw(self,scr):cur_c=self.hov_col if self.is_hov else self.col;sh_off=2;sh_c=(max(0,cur_c[0]-30),max(0,cur_c[1]-30),max(0,cur_c[2]-30));pygame.draw.rect(scr,sh_c,self.rect.move(sh_off,sh_off),border_radius=8);pygame.draw.rect(scr,cur_c,self.rect,border_radius=8);txt_s=self.font.render(self.text,True,BUTTON_TEXT_COLOR);scr.blit(txt_s,txt_s.get_rect(center=self.rect.center))
    def handle_event(self,evt):
        if evt.type==pygame.MOUSEMOTION:self.is_hov=self.rect.collidepoint(evt.pos)
        if evt.type==pygame.MOUSEBUTTONDOWN and self.is_hov and evt.button==1 and self.act:self.act()

def increase_max_speed():
    global global_max_speed, user_requested_speed_change
    global_max_speed+=1.0
    user_requested_speed_change = global_speed_step # Request target speed increase for all cars
    print(f"Max Speed (Increased):{global_max_speed:.1f}")

def decrease_max_speed():
    global global_max_speed,global_min_speed, user_requested_speed_change
    global_max_speed=max(global_min_speed+0.5,global_max_speed-1.0)
    user_requested_speed_change = -global_speed_step # Request target speed decrease
    print(f"Max Speed (Decreased):{global_max_speed:.1f}")

def request_user_quit():
    global user_quit_simulation
    user_quit_simulation=True
    print("Quit requested.")

def run_simulation(genomes,config_neat_obj,screen,clock,sim_globals_param):
    global global_max_speed, user_quit_simulation, user_requested_speed_change

    current_generation_count_local=sim_globals_param["current_generation_count"]
    global_best_fitness_local=sim_globals_param["global_best_fitness"]
    current_generation_count_local+=1
    pygame.display.set_caption(f"NEAT Car Evolution - Gen: {current_generation_count_local}")

    try:
        car_sprite_surface = pygame.image.load(CAR_IMAGE_PATH).convert_alpha()
        car_sprite_surface = pygame.transform.scale(car_sprite_surface, (CAR_SIZE_X, CAR_SIZE_Y))
    except pygame.error as e:
        print(f"ERROR: Car image '{CAR_IMAGE_PATH}' not loaded: {e}"); request_user_quit(); raise UserQuitException()

    nets,cars=[],[]
    for i,(gid,gobj) in enumerate(genomes):
        try:
            new_car = Car(car_sprite_surface)
            if gobj: new_car.genome_key = gobj.key
            nets.append(neat.nn.FeedForwardNetwork.create(gobj,config_neat_obj))
            gobj.fitness=0.0
            cars.append(new_car)
        except Exception as e:
            g_key_str = str(gobj.key) if gobj and hasattr(gobj, 'key') else "N/A"
            print(f"ERROR: NN/Car creation failed for genome {gid} (Key: {g_key_str}): {e}"); continue
    if not cars:
        print("WARNING: No cars created, skipping generation."); sim_globals_param["current_generation_count"]=current_generation_count_local; sim_globals_param["global_best_fitness"]=global_best_fitness_local; return

    try:
        g_map_orig=pygame.image.load('map-d.png').convert_alpha()
        g_map_scaled=pygame.transform.scale(g_map_orig,(GAME_AREA_WIDTH,GAME_AREA_HEIGHT))
    except pygame.error as e: print(f"ERROR: Map not loaded: {e}"); request_user_quit();raise UserQuitException()

    try:
        title_font, info_font, stats_font, button_font = pygame.font.SysFont("Arial",18,True), pygame.font.SysFont("Arial",14), pygame.font.SysFont("Arial",12), pygame.font.SysFont("Arial",14,True)
    except: title_font,info_font,stats_font,button_font = pygame.font.Font(None,24),pygame.font.Font(None,20),pygame.font.Font(None,18),pygame.font.Font(None,20)

    viz_rect=pygame.Rect(VIZ_PANEL_X_OFFSET,VIZ_PANEL_Y_OFFSET,VIZ_PANEL_WIDTH,VIZ_PANEL_HEIGHT)
    stats_panel_x_actual=STATS_PANEL_X_OFFSET
    btn_panel_x = WINDOW_WIDTH - 150 - VIZ_PANEL_X_OFFSET; ui_button_width=140
    ui_elements=[
        Button(btn_panel_x,20,ui_button_width,30,"Max Speed (+)",button_font,BUTTON_COLOR,BUTTON_HOVER_COLOR,increase_max_speed),
        Button(btn_panel_x,60,ui_button_width,30,"Max Speed (-)",button_font,BUTTON_COLOR,BUTTON_HOVER_COLOR,decrease_max_speed),
        Button(btn_panel_x, WINDOW_HEIGHT - 50 - VIZ_PANEL_Y_OFFSET, ui_button_width, 30, "QUIT", button_font, BUTTON_COLOR, BUTTON_HOVER_COLOR, request_user_quit)
    ]
    generation_frames_elapsed, max_frames_per_generation = 0, GENERATION_TIME_LIMIT_SECONDS * FPS
    running_this_generation = True
    best_car_details_current_gen = {"genome": None, "inputs": [], "outputs": [], "chosen_action_idx": -1, "fitness": -float('inf')}

    while running_this_generation:
        if user_quit_simulation: running_this_generation=False; break
        for event in pygame.event.get():
            if event.type==pygame.QUIT: request_user_quit(); running_this_generation=False
            if event.type==pygame.KEYDOWN and event.key==pygame.K_ESCAPE: request_user_quit(); running_this_generation=False
            if event.type==pygame.MOUSEBUTTONDOWN and event.button==1:
                mouse_pos=event.pos; clicked_car_index = -1
                for i_car_click in range(len(cars) - 1, -1, -1):
                    car_clicked_obj = cars[i_car_click]
                    if car_clicked_obj.is_alive() and car_clicked_obj.get_rect_on_screen().collidepoint(mouse_pos): clicked_car_index = i_car_click; break
                if clicked_car_index != -1:
                    car_to_remove = cars[clicked_car_index]
                    if clicked_car_index < len(genomes) and genomes[clicked_car_index] and len(genomes[clicked_car_index]) > 1 and genomes[clicked_car_index][1]:
                         genomes[clicked_car_index][1].fitness = -1000.0
                    print(f"Car index {clicked_car_index} (Genome: {car_to_remove.genome_key}, ID: {car_to_remove.id}) manually removed.")
                    car_to_remove.alive = False
            for ui_el in ui_elements:ui_el.handle_event(event)
        if not running_this_generation: break

        # Apply user requested speed changes globally
        if user_requested_speed_change != 0.0:
            for car_obj in cars:
                if car_obj.is_alive():
                    car_obj.target_speed += user_requested_speed_change
            user_requested_speed_change = 0.0 # Reset request

        total_fitness_this_frame, num_alive_cars_for_avg_fitness = 0.0, 0
        current_gen_best_fitness_val = -float('inf')
        current_gen_best_car_genome_obj, current_gen_best_car_inputs, current_gen_best_car_outputs, current_gen_best_car_action_idx = None, [], [], -1

        for i_car, car_obj in enumerate(cars):
            if car_obj.is_alive():
                nn_input_data=car_obj.get_data_for_nn()
                if i_car < len(nets) and i_car < len(genomes):
                    nn_output_actions=nets[i_car].activate(nn_input_data)
                    car_obj.last_nn_output, car_obj.last_radar_data = list(nn_output_actions), list(nn_input_data)
                    if not nn_output_actions or len(nn_output_actions) != len(ACTION_LABELS): car_obj.alive = False; continue
                    nn_choice_index=nn_output_actions.index(max(nn_output_actions))
                    if nn_choice_index==0: car_obj.target_angle = (car_obj.target_angle + ANGLE_STEP) % 360
                    elif nn_choice_index==1: car_obj.target_angle = (car_obj.target_angle - ANGLE_STEP + 360) % 360
                    elif nn_choice_index==2: car_obj.target_speed -= global_speed_step
                    elif nn_choice_index==3: car_obj.target_speed += global_speed_step
                    car_obj.target_speed = max(global_min_speed, min(car_obj.target_speed, global_max_speed)) # Clamp after NN and global adjustment
                else: car_obj.alive = False; continue
                car_obj.update(g_map_scaled)
                if car_obj.is_alive():
                    current_car_fitness = car_obj.get_fitness()
                    genomes[i_car][1].fitness = current_car_fitness
                    total_fitness_this_frame += current_car_fitness; num_alive_cars_for_avg_fitness += 1
                    if current_car_fitness > current_gen_best_fitness_val:
                        current_gen_best_fitness_val, current_gen_best_car_genome_obj = current_car_fitness, genomes[i_car][1]
                        current_gen_best_car_inputs, current_gen_best_car_outputs, current_gen_best_car_action_idx = list(car_obj.last_radar_data), list(car_obj.last_nn_output), nn_choice_index
                    if current_car_fitness > global_best_fitness_local: global_best_fitness_local = current_car_fitness
        if current_gen_best_car_genome_obj:
             best_car_details_current_gen = {"genome": current_gen_best_car_genome_obj, "inputs": current_gen_best_car_inputs, "outputs": current_gen_best_car_outputs, "chosen_action_idx": current_gen_best_car_action_idx, "fitness": current_gen_best_fitness_val}
        elif num_alive_cars_for_avg_fitness == 0 and generation_frames_elapsed > 0 : best_car_details_current_gen["genome"] = None

        current_alive_cars_count = sum(1 for c in cars if c.is_alive())
        if current_alive_cars_count == 0 and generation_frames_elapsed > FPS : print("No cars left, ending generation."); running_this_generation = False
        generation_frames_elapsed+=1
        if generation_frames_elapsed >= max_frames_per_generation: print(f"Time limit ({GENERATION_TIME_LIMIT_SECONDS}s) reached."); running_this_generation=False

        screen.fill((30,32,44)); screen.blit(g_map_scaled,(GAME_AREA_X_OFFSET,GAME_AREA_Y_OFFSET))
        for car_to_draw in cars:
            if car_to_draw.is_alive():car_to_draw.draw(screen)

        def draw_text_with_background(text_content,font_obj,pos_tuple,surface_obj,pad_x=5,pad_y=2,text_col=None, bg_col=None):
            actual_text_color, actual_bg_color = text_col or TEXT_COLOR, bg_col or INFO_TEXT_BACKGROUND_COLOR
            rendered_text = font_obj.render(text_content,True,actual_text_color)
            bg_rect = rendered_text.get_rect(topleft=pos_tuple); bg_rect.inflate_ip(pad_x*2,pad_y*2)
            shadow_rect = bg_rect.move(1,1); pygame.draw.rect(surface_obj,(0,0,0,max(0, actual_bg_color[3]-150 if len(actual_bg_color)>3 else 50)),shadow_rect,border_radius=5)
            pygame.draw.rect(surface_obj,actual_bg_color,bg_rect,border_radius=5); surface_obj.blit(rendered_text,(pos_tuple[0]+pad_x, pos_tuple[1]+pad_y))
            return rendered_text.get_height() + pad_y*2

        y_offset_info_panel=VIZ_PANEL_Y_OFFSET + 10
        h = draw_text_with_background(f"Gen: {current_generation_count_local}",title_font,(stats_panel_x_actual,y_offset_info_panel),screen);y_offset_info_panel+=h+3
        h = draw_text_with_background(f"Alive: {current_alive_cars_count}/{len(cars)}",info_font,(stats_panel_x_actual,y_offset_info_panel),screen);y_offset_info_panel+=h+1
        h = draw_text_with_background(f"Time: {generation_frames_elapsed//FPS}s / {GENERATION_TIME_LIMIT_SECONDS}s",info_font,(stats_panel_x_actual,y_offset_info_panel),screen);y_offset_info_panel+=h+5
        current_gen_best_fit_display = best_car_details_current_gen["fitness"] if best_car_details_current_gen["fitness"] > -float('inf') else 0.0
        h = draw_text_with_background(f"Gen. Best Fit: {current_gen_best_fit_display:.0f}",stats_font,(stats_panel_x_actual,y_offset_info_panel),screen);y_offset_info_panel+=h+1
        h = draw_text_with_background(f"Global Best Fit: {global_best_fitness_local:.0f}",stats_font,(stats_panel_x_actual,y_offset_info_panel),screen);y_offset_info_panel+=h+1
        avg_fitness_display = total_fitness_this_frame / num_alive_cars_for_avg_fitness if num_alive_cars_for_avg_fitness > 0 else 0.0
        h = draw_text_with_background(f"Gen. Avg. Fit: {avg_fitness_display:.0f}",stats_font,(stats_panel_x_actual,y_offset_info_panel),screen);y_offset_info_panel+=h+3

        best_genome_for_viz = best_car_details_current_gen["genome"]
        if best_genome_for_viz:
            cfg_genome_conf = config_neat_obj.genome_config
            num_inputs, num_outputs = len(cfg_genome_conf.input_keys), len(cfg_genome_conf.output_keys)
            num_hidden = len([nk for nk in best_genome_for_viz.nodes if nk not in cfg_genome_conf.input_keys and nk not in cfg_genome_conf.output_keys])
            active_conns = len([c for c in best_genome_for_viz.connections.values() if c.enabled])
            h = draw_text_with_background("Best Car NN:",stats_font,(stats_panel_x_actual,y_offset_info_panel),screen);y_offset_info_panel+=h+1
            h = draw_text_with_background(f" Inputs: {num_inputs}",stats_font,(stats_panel_x_actual,y_offset_info_panel),screen,2,1);y_offset_info_panel+=h
            h = draw_text_with_background(f" Outputs: {num_outputs}",stats_font,(stats_panel_x_actual,y_offset_info_panel),screen,2,1);y_offset_info_panel+=h
            h = draw_text_with_background(f" Hidden N.: {num_hidden}",stats_font,(stats_panel_x_actual,y_offset_info_panel),screen,2,1);y_offset_info_panel+=h
            h = draw_text_with_background(f" Active Con.: {active_conns}/{len(best_genome_for_viz.connections)}",stats_font,(stats_panel_x_actual,y_offset_info_panel),screen,2,1);y_offset_info_panel+=h
            h = draw_text_with_background(f" Genome ID: {best_genome_for_viz.key}",stats_font,(stats_panel_x_actual,y_offset_info_panel),screen,2,1);y_offset_info_panel+=h+3
            h = draw_text_with_background("Best Car Data:",stats_font,(stats_panel_x_actual,y_offset_info_panel),screen);y_offset_info_panel+=h+1
            viz_inputs, viz_outputs, viz_action_idx = best_car_details_current_gen["inputs"], best_car_details_current_gen["outputs"], best_car_details_current_gen["chosen_action_idx"]
            if viz_inputs: h = draw_text_with_background(" Radars: "+", ".join([f"{v_rad:.2f}" for v_rad in viz_inputs]),stats_font,(stats_panel_x_actual,y_offset_info_panel),screen,2,1);y_offset_info_panel+=h
            if viz_outputs and viz_action_idx != -1 and len(viz_outputs) == len(ACTION_LABELS):
                y_action_text_start=y_offset_info_panel
                for i_text,val_text in enumerate(viz_outputs):
                    lbl_text, text_color_action = ACTION_LABELS[i_text], SELECTED_ACTION_COLOR if i_text==viz_action_idx else TEXT_COLOR
                    h_act_text = draw_text_with_background(f" {lbl_text}: {val_text:.2f}",stats_font,(stats_panel_x_actual,y_action_text_start),screen,2,1,text_color_action); y_action_text_start+=h_act_text
                y_offset_info_panel = y_action_text_start

        max_speed_text_y, max_speed_text_x_coord = 100 + VIZ_PANEL_Y_OFFSET, btn_panel_x + 5
        draw_text_with_background(f"Max Speed:{global_max_speed:.1f}",info_font,(max_speed_text_x_coord, max_speed_text_y),screen)
        for ui_el_draw in ui_elements:ui_el_draw.draw(screen)
        if viz_rect.width>5 and viz_rect.height>5:
            if best_genome_for_viz: draw_neat_visualization(screen,best_genome_for_viz,config_neat_obj,best_car_details_current_gen["inputs"],best_car_details_current_gen["outputs"],viz_rect)
            else:
                empty_viz_surface=pygame.Surface(viz_rect.size,pygame.SRCALPHA); empty_viz_surface.fill(NN_PANEL_BG_COLOR)
                try:font_viz_text=pygame.font.SysFont("Arial",16,True)
                except:font_viz_text=pygame.font.Font(None,22)
                msg_viz_text="Waiting for network...";txt_s_viz_text=font_viz_text.render(msg_viz_text,True,(200,200,220))
                empty_viz_surface.blit(txt_s_viz_text,txt_s_viz_text.get_rect(center=(viz_rect.width//2,viz_rect.height//2))); screen.blit(empty_viz_surface,viz_rect.topleft)
        pygame.display.flip();clock.tick(FPS)

    sim_globals_param["current_generation_count"]=current_generation_count_local
    sim_globals_param["global_best_fitness"]=global_best_fitness_local
    if user_quit_simulation:raise UserQuitException()

if __name__=="__main__":
    pygame.init(); pygame.font.init()
    main_screen, main_clock = pygame.display.set_mode((WINDOW_WIDTH,WINDOW_HEIGHT)), pygame.time.Clock()
    local_dir = os.path.dirname(__file__)
    config_path_abs, car_img_path_abs, map_path_abs = os.path.join(local_dir,CONFIG_PATH), os.path.join(local_dir,CAR_IMAGE_PATH), os.path.join(local_dir, 'map-d.png')
    if not all(os.path.exists(f) for f in [config_path_abs, car_img_path_abs, map_path_abs]):
        print(f"ERROR: Missing files. Config: {config_path_abs}, Car: {car_img_path_abs}, Map: {map_path_abs}"); pygame.quit();sys.exit()
    try: config_neat_main=neat.Config(neat.DefaultGenome,neat.DefaultReproduction,neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path_abs)
    except Exception as e: print(f"ERROR: NEAT config error ({config_path_abs}): {e}"); pygame.quit();sys.exit()

    max_simulation_runs, current_simulation_run_count = 0, 0
    overall_best_genome_ever_across_runs, overall_highest_fitness_ever = None, -float('inf')

    while not user_quit_simulation and (max_simulation_runs==0 or current_simulation_run_count<max_simulation_runs):
        current_simulation_run_count+=1
        print(f"\n\n***** STARTING SIMULATION RUN #{current_simulation_run_count} *****\n")
        population_main=neat.Population(config_neat_main); population_main.add_reporter(neat.StdOutReporter(True)); population_main.add_reporter(neat.StatisticsReporter())
        simulation_globals_dict = {"current_generation_count": 0, "global_best_fitness": overall_highest_fitness_ever if overall_highest_fitness_ever > -float('inf') else 0.0}
        winner_genome_this_run=None
        try:
            max_generations_per_run_limit=100
            winner_genome_this_run=population_main.run(lambda g, c: run_simulation(g, c, main_screen, main_clock, simulation_globals_dict), max_generations_per_run_limit)
            run_actual_best_fitness = simulation_globals_dict["global_best_fitness"]
            if run_actual_best_fitness > overall_highest_fitness_ever:
                 overall_highest_fitness_ever = run_actual_best_fitness; best_genome_candidate = None
                 if winner_genome_this_run and hasattr(winner_genome_this_run, 'fitness') and math.isclose(winner_genome_this_run.fitness, overall_highest_fitness_ever, rel_tol=1e-9): best_genome_candidate = winner_genome_this_run
                 elif population_main.best_genome and hasattr(population_main.best_genome, 'fitness') and math.isclose(population_main.best_genome.fitness, overall_highest_fitness_ever, rel_tol=1e-9): best_genome_candidate = population_main.best_genome
                 if best_genome_candidate: overall_best_genome_ever_across_runs = best_genome_candidate
                 elif population_main.best_genome:
                     if overall_best_genome_ever_across_runs is None or \
                        (hasattr(population_main.best_genome, 'fitness') and hasattr(overall_best_genome_ever_across_runs, 'fitness') and population_main.best_genome.fitness > overall_best_genome_ever_across_runs.fitness) or \
                        (hasattr(population_main.best_genome, 'fitness') and population_main.best_genome.fitness > overall_highest_fitness_ever): # Redundant check, but safe
                        overall_best_genome_ever_across_runs = population_main.best_genome

            if winner_genome_this_run: print(f"\nThreshold met for run #{current_simulation_run_count} (Fitness: {winner_genome_this_run.fitness:.2f}).")
            elif not user_quit_simulation: print(f"\nRun #{current_simulation_run_count} limit reached.")
            if overall_best_genome_ever_across_runs: print(f"Current Overall Best Fitness: {overall_highest_fitness_ever:.2f} (Genome: {overall_best_genome_ever_across_runs.key})")
        except UserQuitException: print("User quit.");user_quit_simulation=True;break
        except (pygame.error, SystemExit) as e: print(f"System/Pygame error: {e}"); user_quit_simulation=True;break
        except Exception as general_err: import traceback; print(f"Unexpected error: {general_err}"); traceback.print_exc(); user_quit_simulation=True;break
        if user_quit_simulation:break

    if overall_best_genome_ever_across_runs: print(f"\n\n***** BEST OVERALL GENOME (Fitness: {overall_highest_fitness_ever:.2f}, ID: {overall_best_genome_ever_across_runs.key}) *****")
    elif not user_quit_simulation: print("\n\nNo genome met threshold or was noteworthy.")
    pygame.quit(); print("Pygame closed. Simulation ended."); sys.exit()