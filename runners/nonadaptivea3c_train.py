from __future__ import division

import time

from datasets.data import get_data
from datasets.glove import Glove

import setproctitle

from models.model_io import ModelOptions

from agents.random_agent import RandomNavigationAgent

import random

from .train_util import (
    compute_loss,
    new_episode,
    run_episode,
    transfer_gradient_from_player_to_shared,
    end_episode,
    reset_player,
)


def nonadaptivea3c_train(
    rank,
    args,
    create_shared_model,
    shared_model,
    initialize_agent,
    optimizer,
    res_queue,
    end_flag,
):
    print('Now Im in nonadaptivea3c_train')
    glove = Glove(args.glove_file)
    scenes, possible_targets, targets = get_data(args.scene_types, args.train_scenes)
    print('We have glove embeddings and data wow')
    random.seed(args.seed + rank)
    idx = [j for j in range(len(args.scene_types))]
    random.shuffle(idx)
    print('scene types have been shuffled')
    setproctitle.setproctitle("Training Agent: {}".format(rank))
    print('some set proctitle bullshit')
    gpu_id = args.gpu_ids[rank % len(args.gpu_ids)]
    print('something gpu id')
    import torch
    print('Torch imported')
    torch.cuda.set_device(gpu_id)
    print('Looks like we cannot set device, cuda is a bunch of fuckers for gpu id : {}'.format(gpu_id))
    torch.manual_seed(args.seed + rank)
    if gpu_id >= 0:
        print('gpu id > 0, who knows what happens next')
        torch.cuda.manual_seed(args.seed + rank)
    print('Done doing gpu bullshit')
    player = initialize_agent(create_shared_model, args, rank, gpu_id=gpu_id)
    print('agent initialized')
    compute_grad = not isinstance(player, RandomNavigationAgent)
    print('Something something compute gradient')
    model_options = ModelOptions()

    j = 0
    print('Right before while loop')
    while not end_flag.value:

        # Get a new episode.
        total_reward = 0
        player.eps_len = 0
        new_episode(
            args, player, scenes[idx[j]], possible_targets, targets[idx[j]], glove=glove
        )
        player_start_time = time.time()

        # Train on the new episode.
        while not player.done:
            # Make sure model is up to date.
            player.sync_with_shared(shared_model)
            # Run episode for num_steps or until player is done.
            total_reward = run_episode(player, args, total_reward, model_options, True)
            # Compute the loss.
            loss = compute_loss(args, player, gpu_id, model_options)
            if compute_grad:
                # Compute gradient.
                player.model.zero_grad()
                loss["total_loss"].backward()
                torch.nn.utils.clip_grad_norm_(player.model.parameters(), 100.0)
                # Transfer gradient to shared model and step optimizer.
                transfer_gradient_from_player_to_shared(player, shared_model, gpu_id)
                optimizer.step()
                # Clear actions and repackage hidden.
            if not player.done:
                reset_player(player)

        for k in loss:
            loss[k] = loss[k].item()

        end_episode(
            player,
            res_queue,
            title=args.scene_types[idx[j]],
            total_time=time.time() - player_start_time,
            total_reward=total_reward,
        )
        reset_player(player)

        j = (j + 1) % len(args.scene_types)
    print('End of while loop')
    player.exit()
