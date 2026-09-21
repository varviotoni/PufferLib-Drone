#pragma once
#include "drone.h"

#define NUM_AVOID_TOWERS 11

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
    float center_tower_radius;
    float circle_radius;
    float collision_penalty;
    float safety_margin;
    float alpha_proximity;
    float target_dist;
    float alpha_dist;
    float alpha_hover;
    int horizon;
} AvoidConfig;

typedef struct {
    TowerObstacle towers[NUM_AVOID_TOWERS]; // 10 towers in circle formation in the middle
    bool* collided;
    float* score;
    float* perf;
    float* collisions;
    float* ema_dist;
    float* ema_vel;
    float* ema_omega;
} AvoidState;

// lifecycle

static void avoid_init(DroneEnv* env) {
    AvoidConfig* cfg = (AvoidConfig*)env->task_config;
    AvoidState* state = (AvoidState*)calloc(1, sizeof(AvoidState));

    float c_radius = (cfg->circle_radius > 0.1f) ? cfg->circle_radius : 3.0f;
    float t_radius = (cfg->tower_radius > 0.05f) ? cfg->tower_radius : 0.45f;

    float center_r = (cfg->center_tower_radius > 0.05f) ? cfg->center_tower_radius : 0.8f;

    // Central tower right in the middle at (0, 0)
    state->towers[0].pos = (Vec3){0.0f, 0.0f, 0.0f};
    state->towers[0].radius = center_r;
    state->towers[0].z_min = -GRID_Z;
    state->towers[0].z_max = GRID_Z;

    // 10 surrounding towers in a circular ring formation
    int num_ring_towers = 10;
    for (int k = 0; k < num_ring_towers; k++) {
        float angle = (2.0f * (float)M_PI * (float)k) / (float)num_ring_towers;
        state->towers[k + 1].pos = (Vec3){
            c_radius * cosf(angle),
            c_radius * sinf(angle),
            0.0f
        };
        state->towers[k + 1].radius = t_radius;
        state->towers[k + 1].z_min = -GRID_Z;
        state->towers[k + 1].z_max = GRID_Z;
    }

    state->collided = (bool*)calloc(env->num_agents, sizeof(bool));
    state->score = (float*)calloc(env->num_agents, sizeof(float));
    state->perf = (float*)calloc(env->num_agents, sizeof(float));
    state->collisions = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_dist = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_vel = (float*)calloc(env->num_agents, sizeof(float));
    state->ema_omega = (float*)calloc(env->num_agents, sizeof(float));
    env->task_state = state;
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

// Reset: spawn drone outside the column ring, goal always on opposite side of column ring
static void avoid_reset(DroneEnv* env, Drone* agent, int idx) {
    AvoidConfig* cfg = (AvoidConfig*)env->task_config;
    AvoidState* state = (AvoidState*)env->task_state;

    float ring_outer = (cfg->circle_radius > 0.1f ? cfg->circle_radius : 3.0f) +
                       (cfg->tower_radius > 0.05f ? cfg->tower_radius : 0.45f);

    // Drone spawns outside the column ring
    float phi = rndf(0.0f, 2.0f * (float)M_PI, &env->rng);
    float r_start = rndf(ring_outer + 0.8f, ring_outer + 2.2f, &env->rng);
    float z_start = rndf(-MARGIN_Z * 0.6f, MARGIN_Z * 0.6f, &env->rng);

    agent->state.pos = (Vec3){
        clampf(r_start * cosf(phi), -MARGIN_X, MARGIN_X),
        clampf(r_start * sinf(phi), -MARGIN_Y, MARGIN_Y),
        clampf(z_start, -MARGIN_Z, MARGIN_Z),
    };

    // Goal is placed at the opposite side of the column ring
    float phi_goal = phi + (float)M_PI + rndf(-0.35f, 0.35f, &env->rng);
    float r_goal = rndf(ring_outer + 0.8f, ring_outer + 2.2f, &env->rng);
    float z_goal = rndf(-MARGIN_Z * 0.6f, MARGIN_Z * 0.6f, &env->rng);

    agent->target->pos = (Vec3){
        clampf(r_goal * cosf(phi_goal), -MARGIN_X, MARGIN_X),
        clampf(r_goal * sinf(phi_goal), -MARGIN_Y, MARGIN_Y),
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

    // Find nearest of the 10 towers in the circle
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
