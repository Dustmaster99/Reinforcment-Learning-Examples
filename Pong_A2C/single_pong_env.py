# simple_pong_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimplePongEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()

        self.width = 800
        self.height = 400

        self.paddle_height = 70
        self.paddle_width = 10
        self.ball_size = 10

        self.agent_x = 30
        self.opponent_x = self.width - 40

        self.paddle_speed = 8
        self.ball_speed_x = 5
        self.ball_speed_y = 4

        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Ações do agente:
        # 0 = parado
        # 1 = subir
        # 2 = descer
        
        self.action_space = spaces.Discrete(3)

        # Estado:
        # [
        #   ball_x,
        #   ball_y,
        #   ball_vx,
        #   ball_vy,
        #   agent_y,
        #   opponent_y,
        #   ball_y - agent_y
        # ]
        
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -1, -1, 0, 0, -1], dtype=np.float32),
            high=np.array([1, 1, 1, 1, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
            )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.ball_x = self.width / 2
        self.ball_y = self.height / 2

        direction = self.np_random.choice([-1, 1])
        self.ball_vx = direction * self.ball_speed_x
        self.ball_vy = self.np_random.uniform(-self.ball_speed_y, self.ball_speed_y)

        self.agent_y = self.height / 2 - self.paddle_height / 2
        self.opponent_y = self.height / 2 - self.paddle_height / 2

        obs = self._get_obs()
        info = {}

        return obs, info

    def step(self, action):
        reward = 0
        terminated = False
        truncated = False

        # -------------------------
        # Movimento do agente
        # -------------------------
        if action == 1:
            self.agent_y -= self.paddle_speed
        elif action == 2:
            self.agent_y += self.paddle_speed

        self.agent_y = np.clip(
            self.agent_y,
            0,
            self.height - self.paddle_height
        )

        # -------------------------
        # Movimento do oponente
        # -------------------------
        self._move_opponent()

        # -------------------------
        # Movimento da bola
        # -------------------------
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Colisão com topo e fundo
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_vy *= -1

        if self.ball_y >= self.height - self.ball_size:
            self.ball_y = self.height - self.ball_size
            self.ball_vy *= -1

        # -------------------------
        # Colisão com raquete do agente
        # -------------------------
        if (
            self.ball_x <= self.agent_x + self.paddle_width
            and self.agent_y <= self.ball_y <= self.agent_y + self.paddle_height
        ):
            self.ball_x = self.agent_x + self.paddle_width
            self.ball_vx *= -1

            relative_hit = (self.ball_y - (self.agent_y + self.paddle_height / 2)) / (self.paddle_height / 2)
            self.ball_vy = relative_hit * 5

        # -------------------------
        # Colisão com raquete do oponente
        # -------------------------
        if (
            self.ball_x + self.ball_size >= self.opponent_x
            and self.opponent_y <= self.ball_y <= self.opponent_y + self.paddle_height
        ):
            self.ball_x = self.opponent_x - self.ball_size
            self.ball_vx *= -1

            relative_hit = (self.ball_y - (self.opponent_y + self.paddle_height / 2)) / (self.paddle_height / 2)
            self.ball_vy = relative_hit * 5

        # -------------------------
        # Pontuação
        # -------------------------
        if self.ball_x < 0:
            reward = -1
            terminated = True

        if self.ball_x > self.width:
            reward = 1
            terminated = True

        obs = self._get_obs()
        info = {}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _move_opponent(self):
        """
        Oponente com inteligência média.

        Ele tenta seguir a bola, mas:
        - só reage quando a bola está indo em sua direção;
        - possui erro aleatório;
        - não se move rápido o suficiente para ser perfeito.
        """

        opponent_center = self.opponent_y + self.paddle_height / 2

        if self.ball_vx > 0:
            target_y = self.ball_y

            # erro proposital para não ser perfeito
            noise = self.np_random.normal(0, 25)
            target_y += noise

            if opponent_center < target_y - 10:
                self.opponent_y += self.paddle_speed * 0.75
            elif opponent_center > target_y + 10:
                self.opponent_y -= self.paddle_speed * 0.75

        else:
            # quando a bola vai para o agente, o oponente volta lentamente ao centro
            center_target = self.height / 2

            if opponent_center < center_target - 10:
                self.opponent_y += self.paddle_speed * 0.4
            elif opponent_center > center_target + 10:
                self.opponent_y -= self.paddle_speed * 0.4

        self.opponent_y = np.clip(
            self.opponent_y,
            0,
            self.height - self.paddle_height
        )

    def _get_obs(self):
        return np.array([
            self.ball_x / self.width,
            self.ball_y / self.height,
            self.ball_vx / 10,
            self.ball_vy / 10,
            self.agent_y / self.height,
            self.opponent_y / self.height,
            (self.ball_y - self.agent_y) / self.height
        ], dtype=np.float32)

    def render(self):
        import pygame

        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Simple Pong RL")

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.fill((0, 0, 0))

        pygame.draw.rect(
            self.window,
            (255, 255, 255),
            pygame.Rect(
                self.agent_x,
                self.agent_y,
                self.paddle_width,
                self.paddle_height
            )
        )

        pygame.draw.rect(
            self.window,
            (255, 255, 255),
            pygame.Rect(
                self.opponent_x,
                self.opponent_y,
                self.paddle_width,
                self.paddle_height
            )
        )

        pygame.draw.rect(
            self.window,
            (255, 255, 255),
            pygame.Rect(
                self.ball_x,
                self.ball_y,
                self.ball_size,
                self.ball_size
            )
        )

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None