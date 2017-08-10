## Imitation Learning with Dataset Aggregation (DAGGER) on Torcs Env

This repository implements a simple algorithm for imitation learning: [DAGGER](https://www.cs.cmu.edu/~sross1/publications/Ross-AIStats11-NoRegret.pdf).
In this example, the agent only learns to control the steer [-1, 1], the speed is computed
automatically in `gym_torcs.TorcsEnv`.

## Requirements

1. Ubuntu (I only test on this)
2. Python 3
3. TensorLayer and TensorFlow 
4. [Gym-Torcs](https://github.com/ugo-nama-kun/gym_torcs)

## Setting Up

It is a little bit boring to set up the environment, but any incorrect configurations will lead to FAILURE.
After installing [Gym-Torcs](https://github.com/ugo-nama-kun/gym_torcs), please follow the instructions to confirm everything work well:

- Open a terminal:
  - Run `sudo torcs -vision` to start a game
  - `Race --> Practice --> Configure Race`: set the driver to `scr_server 1` instead of `player`
  - Open Torcs server by selecting `Race --> Practice --> New Race`:
This should result that Torcs keeps a blue screen with several text information.

- Open another terminal:
  - Run `python snakeoil3_gym.py` on another terminal, it will shows how the fake AI control the car.
  - Press F2 to see the driver view.

- Set image size to 64x64x3:
  - The model is trained on 64x64 RGB observation.
  - Run `sudo torcs -vision` to start a game
  - `Options --> Display --> select 64x64 --> Apply`


## Usage
Make sure everything above work well and then run:

- `python dagger.py`

It will start a Torcs server at the beginning of every episode, and terminate the server when the car crashs or the speed is too low.
Note that, the self-contained `gym_torcs.py` is modified from [Gym-Torcs](https://github.com/ugo-nama-kun/gym_torcs), you can try different settings (like default speed, terminated speed) by modifying it.

## Results 

After Episode 1, the car crashes after 315 steps. 

![](http://i.imgur.com/YfqFXQZ.gif)

<!---
After Episode 2, crashes after 151 steps

![](http://i.imgur.com/0bXKyVx.gif)


After Episode 3, crashes after 395 steps

![](http://i.imgur.com/doz8U0z.gif)

After Episode 4, the car does not crash anymore: [gif](http://i.imgur.com/pKeVxLY.gif).
-->

After Episode 3, the car does not crash anymore !!!

![](http://i.imgur.com/doz8U0z.gif)

The number of steps and episodes might vary depending on the parameters initialization.


ENJOY !

<!---
Note: The images fed to the model are 64x64, the images shown above have been resized to 256x256 for viewing purposes.
-->
