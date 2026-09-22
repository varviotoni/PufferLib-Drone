#pragma once
#include "drone.h"

#define NUM_AVOID_TOWERS 26

#define AVOID_SCORE_DIST_SCALE 0.01f
#define AVOID_SCORE_VEL_SCALE 0.01f
#define AVOID_SCORE_OMEGA_SCALE 0.01f

typedef struct {
    Vec3 pos;       // Center (x, y, 0)
    float radius;   // Radius of cylinder
    float z_min;    // Floor (-GRID_Z)
    float z_max;    // Ceiling (GRID_Z)
} TowerObstacle;

typedef struct {
    float tower_radius;
    float line_spacing;
    float tower_spacing;
    float collision_penalty;
    float safety_margin;
    float alpha_proximity;
    float target_dist;
    float alpha_dist;
    float alpha_hover;
    int horizon;
} AvoidConfig;

typedef struct {
    TowerObstacle towers[NUM_AVOID_TOWERS]; // 4 staggered lines of towers in the middle (7-6-7-6)
    bool* collided;
    float* score;
    float* perf;
    float* collisions;
    float* ema_dist;
    float* ema_vel;
    float* ema_omega;
} AvoidState;

// lifecycle

static void avoid_generate_towers(DroneEnv* env) {
    AvoidConfig* cfg = (AvoidConfig*)env->task_config;
    AvoidState* state = (AvoidState*)env->task_state;

    float x_dist = (cfg->line_spacing > 0.1f) ? cfg->line_spacing : 2.5f;
    float y_dist = (cfg->tower_spacing > 0.1f) ? cfg->tower_spacing : 3.0f;
    float base_radius = (cfg->tower_radius > 0.05f) ? cfg->tower_radius : 0.45f;

    // Adaptive jitter along Y to ensure adjacent towers always have clearance
    float max_yjitter = fmaxf(0.05f, (y_dist - 2.0f * base_radius) * 0.35f);
    float max_xjitter = 0.35f;

    int idx = 0;

    // Line 1: 7 towers at X ≈ -x_dist (randomized X wobble and Y position)
    for (int j = -3; j <= 3; j++) {
        float x = -x_dist + rndf(-max_xjitter, max_xjitter, &env->rng);
        float y;
        if (j == -3) {
            y = -MARGIN_Y + rndf(0.0f, 0.2f, &env->rng);
        } else if (j == 3) {
            y = MARGIN_Y - rndf(0.0f, 0.2f, &env->rng);
        } else {
            y = (float)j * y_dist + rndf(-max_yjitter, max_yjitter, &env->rng);
        }
        float r = base_radius * rndf(0.9f, 1.1f, &env->rng);
        state->towers[idx].pos = (Vec3){x, clampf(y, -MARGIN_Y, MARGIN_Y), 0.0f};
        state->towers[idx].radius = r;
        state->towers[idx].z_min = -GRID_Z;
        state->towers[idx].z_max = GRID_Z;
        idx++;
    }

    // Line 2: 6 towers at X ≈ 0.0, Y staggered by y_dist / 2 (randomized)
    for (int j = -2; j <= 3; j++) {
        float x = rndf(-max_xjitter, max_xjitter, &env->rng);
        float y = ((float)j - 0.5f) * y_dist + rndf(-max_yjitter, max_yjitter, &env->rng);
        float r = base_radius * rndf(0.9f, 1.1f, &env->rng);
        state->towers[idx].pos = (Vec3){x, clampf(y, -MARGIN_Y, MARGIN_Y), 0.0f};
        state->towers[idx].radius = r;
        state->towers[idx].z_min = -GRID_Z;
        state->towers[idx].z_max = GRID_Z;
        idx++;
    }

    // Line 3: 7 towers at X ≈ +x_dist (randomized)
    for (int j = -3; j <= 3; j++) {
        float x = x_dist + rndf(-max_xjitter, max_xjitter, &env->rng);
        float y;
        if (j == -3) {
            y = -MARGIN_Y + rndf(0.0f, 0.2f, &env->rng);
        } else if (j == 3) {
            y = MARGIN_Y - rndf(0.0f, 0.2f, &env->rng);
        } else {
            y = (float)j * y_dist + rndf(-max_yjitter, max_yjitter, &env->rng);
        }
        float r = base_radius * rndf(0.9f, 1.1f, &env->rng);
        state->towers[idx].pos = (Vec3){x, clampf(y, -MARGIN_Y, MARGIN_Y), 0.0f};
        state->towers[idx].radius = r;
        state->towers[idx].z_min = -GRID_Z;
        state->towers[idx].z_max = GRID_Z;
        idx++;
    }

    // Line 4: 6 towers at X ≈ +2.0*x_dist, Y staggered by y_dist / 2 (randomized)
    for (int j = -2; j <= 3; j++) {
        float x = 2.0f * x_dist + rndf(-max_xjitter, max_xjitter, &env->rng);
        float y = ((float)j - 0.5f) * y_dist + rndf(-max_yjitter, max_yjitter, &env->rng);
        float r = base_radius * rndf(0.9f, 1.1f, &env->rng);
        state->towers[idx].pos = (Vec3){x, clampf(y, -MARGIN_Y, MARGIN_Y), 0.0f};
        state->towers[idx].radius = r;
        state->towers[idx].z_min = -GRID_Z;
        state->towers[idx].z_max = GRID_Z;
        idx++;
    }
}

static void avoid_env_reset(DroneEnv* env) {
    avoid_generate_towers(env);
}

static void avoid_init(DroneEnv* env) {
    AvoidState* state = (AvoidState*)calloc(1, sizeof(AvoidState));
    env->task_state = state;

    avoid_generate_towers(env);

    state->collided = (bool*)calloc(env->num_agents, sizeof(bool));
    state->score = (float*)calloc(env->num_agents, sizeof(float));
    state->perf = (float*)calloc(env->num_agents, sizeof(float));
    state->collisions = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_dist = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_vel = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_omega = (float*)calloc(env->num_agents, sizeof(float));
}

static void avoid_close(DroneEnv* env) {
    AvoidState* state = (AvoidState*)env->task_state;
    if (state != NULL) {
        free(state->collided);
        free(state->score);
        free(state->perf);
        free(state->collisions);
        free(state->ema_dist);
        free(state->ema_vel);
        free(state->ema_omega);
        free(state);
    }
    free(env->task_config);
}

// Reset: spawn drone on one side of the lines, goal on opposite side of the lines
static void avoid_reset(DroneEnv* env, Drone* agent, int idx) {
    AvoidConfig* cfg = (AvoidConfig*)env->task_config;
    AvoidState* state = (AvoidState*)env->task_state;

    if (env->num_agents == 1) {
        avoid_generate_towers(env);
    }

    float x_dist = (cfg->line_spacing > 0.1f) ? cfg->line_spacing : 2.5f;

    // Randomize travel direction: 50% left-to-right (-X to +X), 50% right-to-left (+X to -X)
    bool left_to_right = (rndf(0.0f, 1.0f, &env->rng) < 0.5f);

    // Negative side: before Line 1 (at -x_dist)
    float x_neg = rndf(-MARGIN_X + 1.0f, -x_dist - 2.5f, &env->rng);
    // Positive side: after Line 4 (at +2.0*x_dist)
    float x_pos = rndf(2.0f * x_dist + 1.8f, MARGIN_X - 0.8f, &env->rng);

    float x_start = left_to_right ? x_neg : x_pos;
    float x_goal  = left_to_right ? x_pos : x_neg;

    // Y spans across the obstacle channel [-4.5m, 4.5m]
    float y_start = rndf(-4.5f, 4.5f, &env->rng);
    float y_goal  = rndf(-4.5f, 4.5f, &env->rng);

    // Z within normal arena flight height
    float z_start = rndf(-MARGIN_Z * 0.5f, MARGIN_Z * 0.5f, &env->rng);
    float z_goal  = rndf(-MARGIN_Z * 0.5f, MARGIN_Z * 0.5f, &env->rng);

    agent->state.pos = (Vec3){
        clampf(x_start, -MARGIN_X, MARGIN_X),
        clampf(y_start, -MARGIN_Y, MARGIN_Y),
        clampf(z_start, -MARGIN_Z, MARGIN_Z),
    };

    // Face towards the goal
    agent->state.quat = left_to_right ? (Quat){1.0f, 0.0f, 0.0f, 0.0f} : (Quat){0.0f, 0.0f, 0.0f, 1.0f};

    agent->target->pos = (Vec3){
        clampf(x_goal, -MARGIN_X, MARGIN_X),
        clampf(y_goal, -MARGIN_Y, MARGIN_Y),
        clampf(z_goal, -MARGIN_Z, MARGIN_Z),
    };

    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
    agent->target->normal = (Vec3){0.0f, 0.0f, 0.0f};
    state->collided[idx] = false;

    float dist = norm3(sub3(agent->target->pos, agent->state.pos));
    float vel = norm3(agent->state.vel);
    float omega = norm3(agent->state.omega);
    state->score[idx] = 0.0f;
    state->perf[idx] = hover_score(dist, vel, omega);
    state->ema_dist[idx] = dist;
    state->ema_vel[idx] = vel;
    state->ema_omega[idx] = omega;
}

// Reward: distance progress + hover score - proximity penalty - collision penalty
static float avoid_reward(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    AvoidConfig* cfg = (AvoidConfig*)env->task_config;
    AvoidState* state = (AvoidState*)env->task_state;

    // Find nearest of the 26 towers in the 4 staggered lines
    float min_d_xy = 1e9f;
    int closest = 0;
    for (int k = 0; k < NUM_AVOID_TOWERS; k++) {
        float d = hypotf(agent->state.pos.x - state->towers[k].pos.x,
                         agent->state.pos.y - state->towers[k].pos.y);
        if (d < min_d_xy) {
            min_d_xy = d;
            closest = k;
        }
    }

    float drone_radius = 0.15f;
    float collision_dist = state->towers[closest].radius + drone_radius;

    float reward = 0.0f;

    // Check collision with any tower
    if (min_d_xy <= collision_dist) {
        state->collided[idx] = true;
        state->collisions[idx] += 1.0f;
        reward -= cfg->collision_penalty;
    } else {
        // Soft proximity penalty if entering safety margin of nearest tower
        float safe_zone = collision_dist + cfg->safety_margin;
        if (min_d_xy < safe_zone && cfg->safety_margin > 0.001f) {
            float pen = (safe_zone - min_d_xy) / cfg->safety_margin;
            reward -= cfg->alpha_proximity * pen * pen;
        }

        // Distance progress to goal
        reward += cfg->alpha_dist * (cache->prev_dist - cache->dist);

        // Hover stabilization reward at goal
        float score = hover_score(cache->dist, cache->vel, cache->omega);
        reward += cfg->alpha_hover * score;

        state->score[idx] += score;
        state->perf[idx] = 0.98f * state->perf[idx] + 0.02f * score;
    }

    state->ema_dist[idx] = 0.99f * state->ema_dist[idx] + 0.01f * cache->dist;
    state->ema_vel[idx] = 0.99f * state->ema_vel[idx] + 0.01f * cache->vel;
    state->ema_omega[idx] = 0.99f * state->ema_omega[idx] + 0.01f * cache->omega;

    if (out_of_bounds(agent->state.pos, 1.0f)) {
        reward -= env->oob_penalty;
    }

    return reward;
}

// Done: immediately terminate on collision, OOB, or horizon limit
static bool avoid_done(DroneEnv* env, Drone* agent, int idx, StepCache* cache) {
    AvoidConfig* cfg = (AvoidConfig*)env->task_config;
    AvoidState* state = (AvoidState*)env->task_state;

    if (state->collided[idx]) {
        return true;
    }

    return out_of_bounds(agent->state.pos, 1.0f) || agent->episode_length >= cfg->horizon;
}

static void avoid_log(DroneEnv* env, Drone* agent, int idx, Log* log, StepCache* cache) {
    AvoidState* state = (AvoidState*)env->task_state;
    TaskLog* t = &log->task[env->task];
    t->n += 1.0f;
    t->perf += state->perf[idx];
    t->score += state->score[idx];
    t->keys[0] += state->ema_dist[idx];
    t->keys[1] += state->ema_vel[idx];
    t->keys[2] += state->ema_omega[idx];
    t->keys[3] += state->collided[idx] ? 1.0f : 0.0f;
}
