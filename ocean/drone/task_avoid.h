#pragma once
#include "drone.h"

#define AVOID_SCORE_DIST_SCALE 0.01f
#define AVOID_SCORE_VEL_SCALE 0.01f
#define AVOID_SCORE_OMEGA_SCALE 0.01f

typedef struct {
    Vec3 pos;       // Center (x, y, 0)
    float radius;   // Radius of cylinder
    float z_min;    // Floor (-MARGIN_Z or -GRID_Z)
    float z_max;    // Ceiling (MARGIN_Z or GRID_Z)
} TowerObstacle;

typedef struct {
    float tower_radius;
    float collision_penalty;
    float safety_margin;
    float alpha_proximity;
    float target_dist;
    float alpha_dist;
    float alpha_hover;
    int horizon;
} AvoidConfig;

typedef struct {
    TowerObstacle* towers; // 1 tower per agent: towers[idx]
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
    AvoidState* state = (AvoidState*)calloc(1, sizeof(AvoidState));
    state->towers = (TowerObstacle*)calloc(env->num_agents, sizeof(TowerObstacle));
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
        free(state->towers);
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

// Reset: place goal, start drone, and place full-height tower obstacle in between
static void avoid_reset(DroneEnv* env, Drone* agent, int idx) {
    AvoidConfig* cfg = (AvoidConfig*)env->task_config;
    AvoidState* state = (AvoidState*)env->task_state;

    Vec3 target = random_pos(&env->rng);
    agent->target->pos = target;
    agent->target->vel = (Vec3){0.0f, 0.0f, 0.0f};
    agent->target->normal = (Vec3){0.0f, 0.0f, 0.0f};

    // Starting position of drone at distance cfg->target_dist from target
    Vec3 p = add3(target, random_ball_offset(&env->rng, cfg->target_dist));
    agent->state.pos = (Vec3){
        clampf(p.x, -MARGIN_X, MARGIN_X),
        clampf(p.y, -MARGIN_Y, MARGIN_Y),
        clampf(p.z, -MARGIN_Z, MARGIN_Z),
    };

    // Place full-height tower obstacle between start and target
    Vec3 start = agent->state.pos;
    Vec3 goal = target;

    // Segment midpoint with randomization
    float alpha = rndf(0.35f, 0.65f, &env->rng);
    Vec3 mid = (Vec3){
        start.x + alpha * (goal.x - start.x),
        start.y + alpha * (goal.y - start.y),
        0.0f
    };

    // Add lateral jitter perpendicular to flight trajectory
    float dx = goal.x - start.x;
    float dy = goal.y - start.y;
    float len_xy = sqrtf(dx * dx + dy * dy);
    if (len_xy > 0.01f) {
        float perp_x = -dy / len_xy;
        float perp_y = dx / len_xy;
        float jitter = rndf(-0.5f, 0.5f, &env->rng);
        mid.x += perp_x * jitter;
        mid.y += perp_y * jitter;
    }

    mid.x = clampf(mid.x, -MARGIN_X, MARGIN_X);
    mid.y = clampf(mid.y, -MARGIN_Y, MARGIN_Y);

    state->towers[idx].pos = (Vec3){mid.x, mid.y, 0.0f};
    state->towers[idx].radius = cfg->tower_radius > 0.0f ? cfg->tower_radius : 0.5f;
    state->towers[idx].z_min = -GRID_Z;
    state->towers[idx].z_max = GRID_Z;
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
    TowerObstacle* tower = &state->towers[idx];

    // Distance in XY plane to vertical cylinder centerline
    float d_xy = hypotf(agent->state.pos.x - tower->pos.x, agent->state.pos.y - tower->pos.y);
    float drone_radius = 0.15f;
    float collision_dist = tower->radius + drone_radius;

    float reward = 0.0f;

    // Check collision
    if (d_xy <= collision_dist) {
        state->collided[idx] = true;
        state->collisions[idx] += 1.0f;
        reward -= cfg->collision_penalty;
    } else {
        // Soft proximity penalty when entering safety zone
        float safe_zone = collision_dist + cfg->safety_margin;
        if (d_xy < safe_zone && cfg->safety_margin > 0.001f) {
            float pen = (safe_zone - d_xy) / cfg->safety_margin;
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

    if (cache->dist > cfg->target_dist + 1.5f) {
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

    return cache->dist > (cfg->target_dist + 1.5f) || agent->episode_length >= cfg->horizon;
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
