from src.module.context import Profile as P
from src.util.tools import IO, Logger, Funcs


class Test:
    @staticmethod
    def plot_maze():
        from src.module.env.maze import Maze
        env = Maze.make_env(True)
        env.reset()
        env.render()  # plot the maze
        input()

    @staticmethod
    # def build_graph_test_auto(a=2, n=5, s=1, e=1, m=3, seed_range=[0, 20]):
    def build_graph_test_auto(a=5, n=50, s=5, e=7, m=30, seed_range=[0, 300]):
        """
        a: num actions
        n: total sates
        s: starting states
        e: ending states
        m: num traj.

        this section test the shrunk grpah building algorithm:
        1. auto gen traj.
        1.1 assign n states (s start, and e end states, a actions)
        1.2 random the list of the set
        1.3 select the valid sub-sequence

        2. sanity check of the built shrunk graph
        2.1 build the graph using m traj.s
        2.2 test the connectivities in the traj.s on the graph
        """
        from src.module.agent.memory.memory import Memory
        import random

        for seed in range(seed_range[0], seed_range[1]):
            random.seed(seed)
            print(f"seed: {seed}")

            IO.renew_dir(P.result_dir)

            memory = Memory(id=0, is_head=True)
            states = list(range(n))
            actions = list(range(a))

            starting_states = set()
            while True:
                if len(starting_states) < s:
                    starting_states.add(random.choice(states))
                else:
                    break

            ending_states = set()
            while True:
                if len(ending_states) < e:
                    end_state = random.choice(states)
                    if end_state not in starting_states:
                        ending_states.add(end_state)
                else:
                    break

            for _ in range(m):
                traj = []
                cur_state = random.choice(list(starting_states))
                cur_reward = 0
                total_reward = 0
                while True:
                    # cur_action = random.choice(actions)
                    next_state = random.choice(states)  # could be any state but ending_states
                    cur_action = next_state - cur_state 
                    traj.append([cur_state, cur_action, next_state, cur_reward])
                    cur_state = next_state
                    cur_reward = 0  # set all states zero-reward
                    total_reward += cur_reward

                    if next_state in ending_states:
                        break
                memory.store_inc(traj, total_reward)
                # print(traj)
        
            memory.merge_inc(memory.inc)
            memory.post_process()

    @staticmethod
    def build_graph_test_manual():

        from src.module.agent.memory.memory import Memory


        IO.renew_dir(P.result_dir)
        a = Memory(0, True)

        classic_case = "classic"
        special_case = "special"
        case = classic_case

        # classic case: three traj. meet
        if case == classic_case:
            traj1, traj1_tr = [
                ["a0", 1, "a1", 0],
                ["a1", 1, "a2", 0],
                ["a2", 1, "a3", 0],
                ["a3", 1, "a4", 4],
                ["a4", 1, "a5", 0],
                ["a5", 1, "a6", 5],
            ], 9
            traj2, traj2_tr = [
                ["b0", 2, "b1", 0],
                ["b1", 2, "a2", 0],
                ["a2", 2, "b2", 0],
                ["b2", 2, "b3", 2],
                ["b3", 2, "b4", 0],
                ["b4", 2, "b5", 0],
                ["b5", 2, "b6", 0],
            ], 2
            traj3, traj3_tr = [
                ["c0", 3, "c1", 0],
                ["c1", 3, "b5", 0],
                ["b5", 3, "c2", 0],
                ["c2", 3, "c3", 0],
                ["c3", 3, "a4", 4],
                ["a4", 3, "c4", 0],
                ["c4", 3, "c5", 3],
            ], 3

            a.store_inc(traj1, traj1_tr)
            a.store_inc(traj2, traj2_tr)
            a.store_inc(traj3, traj3_tr)

            important_obs = ["a2", "b5", "a4"]

        # special cases
        if case == special_case:
            traj4, traj4_tr = [
                ["d0", 4, "d1", 1],
                ["d1", 4, "d2", 1],
                ["d2", 4, "d3", 1],
                ["d3", 4, "d4", 1],
                ["d4", 4, "d5", 1],
                ["d5", 4, "d6", 1],
                ["d6", 4, "d7", 1],
                ["d7", 4, "d8", 1],
                ["d8", 4, "d9", 1],
                ["d9", 4, "d10", 5],
            ], 14
            traj5, traj5_tr = [
                ["d0", 4, "d1", 1],
                ["d1", 4, "d2", 1],
                ["d2", 4, "d3", 1],
                ["d3", 4, "d4", 1],
                ["d4", 4, "d5", 1],
                ["d5", 4, "d6", 1],
                ["d6", 4, "d7", 1],
                ["d7", 6, "d11", 1],
                ["d11", 6, "d12", 1],
                ["d12", 6, "d13", 20],
            ], 14
            traj6, traj6_tr = [
                ["d0", 4, "d1", 1],
                ["d1", 4, "d3", 1],
                ["d3", 4, "d4", 1],
                ["d4", 4, "d5", 1],
                ["d5", 4, "d6", 1],
                ["d6", 4, "d7", 1],
                ["d7", 4, "d8", 1],
                ["d8", 4, "d9", 1],
                ["d9", 4, "d10", 5],
            ], 14

            a.store_inc(traj4, traj4_tr)  # loop
            a.store_inc(traj5, traj5_tr)  # loop
            a.store_inc(traj6, traj6_tr)  # loop

            important_obs = ["d7"]

        # merge
        a.merge_inc(a.inc)

        a.post_process()
        print("------------------------")
        for obs in important_obs:
            print(a.get_action(obs))  # should be 2, 3, 1 for classic case

    @staticmethod 
    def vi_test():
        from src.module.agent.memory.iterator import Iterator
        import numpy as np

        iterator = Iterator(0)
        n = 10

        np.random.seed(1)
        np_adj = np.random.random([n, n])
        np.random.seed(2)
        np_rew = np.random.random([n])
        np.random.seed(3)
        np_val_0 = np.random.random([n])

        np_adj = np.int8(np.round(np_adj))
        np_rew = np.float32(np_rew)
        np_val_0 = np.float32(np_val_0)

        iterator.iterate(np_adj, np_rew, np_val_0)

    @staticmethod
    def maze_vi_validation():
        import time

        model = IO.read_disk_dump(f"{P.model_dir}maze_original-optimal.pkl")
        max_total_reward_init_obs = model.general_info["max_total_reward_init_obs"]
        print(max_total_reward_init_obs)
        max_total_reward_traj = model.general_info["max_total_reward_traj"]
        from src.module.env.maze import Maze
        env = Maze.make_env(True)
        env.reset()
        done = False
        total_reward = 0
        model.draw_graph()
        for step, [last_obs, prev_action, obs, reward] in enumerate(max_total_reward_traj, start=1):
            if last_obs in model.obs_best_action:
                searched_action = model.obs_best_action[last_obs]
            real_obs, reward, done, info = env.step(prev_action)
            assert obs == real_obs, "error"
            total_reward += reward
            if done:
                print(total_reward)
                break
            else:
                print(f"#{step} {real_obs} -{searched_action}-{prev_action}- {reward}")
            env.render()
            time.sleep(0.01)
        env.render()

        breakpoint()

    @staticmethod
    def sokoban_vi_validation():
        actions = {
            0: "^",
            1: "v",
            2: "<",
            3: ">",
        }
        print(actions)

        def formated_obs(o):
            s = ""
            for i in range(len(o)):
                if i > 0 and i % 10 == 0:
                    s += "\n"
                s += o[i]
            s += "\n"
            return s

        model = IO.read_disk_dump(f"{P.model_dir}Boxoban-Val-v1-optimal.pkl")
        max_total_reward_init_obs = model.general_info["max_total_reward_init_obs"]
        max_total_reward = model.general_info["max_total_reward"]
        print(max_total_reward)
        print(formated_obs(max_total_reward_init_obs))
        max_total_reward_traj = model.general_info["max_total_reward_traj"]
        from src.module.env.sokoban import Sokoban
        env = Sokoban.make_env()
        env.reset()
        done = False
        total_reward = 0
        for step, [last_obs, prev_action, obs, reward] in enumerate(max_total_reward_traj, start=1):
            if last_obs in model.obs_best_action:
                searched_action = model.obs_best_action[last_obs]
            real_obs, reward, done, info = env.step(prev_action)
            if obs != real_obs:
                print(f"error ({step}): \n{formated_obs(last_obs)}action: {actions[prev_action]}\n{formated_obs(obs)}&\n{formated_obs(real_obs)}")
            total_reward += reward
            if done:
                print(total_reward)
                break
            else:
                pass
                print(f"#{step} \n{formated_obs(real_obs)} -{actions[searched_action]}-{actions[prev_action]}- \n{formated_obs(obs)} {reward}\n")

        breakpoint()

    @staticmethod 
    def hashing_test():
        import numpy as np

        # max resolution of hashing is 1e-16
        a = np.random.random([1, 500])
        print(a[0, -1])
        print(Funcs.matrix_hashing(a))
        a[0, -1] += 1e-16
        print(a[0, -1])
        print(Funcs.matrix_hashing(a))

        # not work
        a = np.random.random([1, 500])
        print(a[0, -1])
        print(Funcs.matrix_hashing(a))
        a[0, -1] += 1e-17
        print(a[0, -1])
        print(Funcs.matrix_hashing(a))

    @staticmethod
    def make_atari_alternative_env():
        from src.module.env.atari_alternative import Atari

        env = Atari.make_env()
        env.reset()
        for i in range(10):
            obs, reward, done, info = env.step(i)
            print(f"{obs.shape} {obs.dtype}")
            print(reward)
            if done:
                break
    
    @staticmethod
    def atari_play():
        import gym
        from gym.utils.play import play
        
        env_name = P.env_name_list[24]
        Logger.log(f"env name: {env_name}")
        env = gym.make(f"{env_name}NoFrameskip-v4")
        if hasattr(env, "get_keys_to_action"):
            keys_to_action = env.get_keys_to_action()
        elif hasattr(env.unwrapped, "get_keys_to_action"):
            keys_to_action = env.unwrapped.get_keys_to_action()
        for item in keys_to_action:
            Logger.log(f"key map: {[chr(key) for key in item]} -> {keys_to_action[item]}")

        play(env, zoom=4)

    @staticmethod
    def nrnn():
        from src.module.agent.memory.projector import NRNNProjector
        import numpy as np
        projector = NRNNProjector(id, True)

        last_obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=1.)
        obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=1.)
        a, b = projector.batch_project([last_obs, obs])
        print(f"{a}  {b}")

        last_obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=2.)
        obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=3.)
        a, b = projector.batch_project([last_obs, obs])
        print(f"{a}  {b}")

        last_obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=2.)
        obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=3.)
        a, b = projector.batch_project([last_obs, obs])
        print(f"{a}  {b}")

        last_obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=2.)
        obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=3.)
        a, b = projector.batch_project([last_obs, obs])
        print(f"{a}  {b}")

        last_obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=2.)
        obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=3.)
        a, b = projector.batch_project([last_obs, obs])
        print(f"{a}  {b}")

    @staticmethod
    def rnn():
        from src.module.agent.memory.projector import RNNProjector, HashProjector
        import numpy as np
        # projector = HashProjector(id)
        projector = RNNProjector(id, True)
        
        last_obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=1.)
        obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=2.)
        a, b = projector.batch_project([last_obs, obs])

        last_obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=3.)
        obs = np.full(shape=[P.screen_size * P.screen_size * P.stack_frames], fill_value=4.)
        c, d = projector.batch_project([last_obs, obs])

        print(f"a {a}\nb {b}\n\nc {c}\nd {d}")

    @staticmethod
    def football():
        from src.module.env.football import Football

        env = Football.make_env()

        env .reset()

        breakpoint()

    @staticmethod
    def test_pypullet():
        import pybullet as p
        import time
        import pybullet_data
        physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
        p.setGravity(0, 0, -10)
        planeId = p.loadURDF("plane.urdf")
        startPos = [0, 0, 1]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        boxId = p.loadURDF("r2d2.urdf", startPos, startOrientation)
        # set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
        for i in range(10000):
            p.stepSimulation()
            time.sleep(1. / 240.)
        cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
        print(cubePos, cubeOrn)
        p.disconnect()

    @staticmethod
    def test_sokoban():
        import gym
        import gym_sokoban

        env = gym.make("Sokoban-v0")
        env.reset()
        print({
            "no op": 0,
            "push ^": 1,
            "push v": 2,
            "push <": 3,
            "push >": 4,
            "^": 5,
            "v": 6,
            "<": 7,
            ">": 8,
        })
        done = False
        reward = 0
        total_reward = 0
        while not done:
            env.render(mode="human")
            action = env.action_space.sample()
            # while True:
            #     try:
            #         action = int(input(f"total reward: {total_reward} action: "))
            #         if action not in range(9):
            #             raise Exception()
            #         break
            #     except Exception:
            #         pass
            obs, reward, done, info = env.step(action)
            breakpoint()
            total_reward += reward
        Logger.log(f"total reward: {total_reward}")

    @staticmethod
    def test_tiny_sokoban():
        from src.module.env.sokoban import Sokoban

        env = Sokoban.make_env()
        env.reset()
        done = False
        print({
            "^": 0,
            "v": 1,
            "<": 2,
            ">": 3,
        })
        while not done:
            env.render(mode="human")
            action = env.action_space.sample()
            while True:
                try:
                    action = int(input("action: "))
                    if action not in range(9):
                        raise Exception()
                    break
                except Exception:
                    pass
            obs, reward, done, info = env.step(action)
            # env.reset()


    @staticmethod
    def read_highway_graph():
        from src.module.agent.memory.storage import Graph
        import numpy as np

        ms = IO.read_disk_dump(f"{P.model_dir}Boxoban-Test-v0-optimal.pkl")
        print(len(ms._node))

        def po(nid):
            print("\n")
            if nid in ms._crossing_nodes:
                print(f"crossing_node {nid}:")
            else:
                print(f"shrunk_node {nid}:")

            o = ms._node[nid][Graph._node_obs][0]
            s = ""
            for i in range(len(o)):
                if i > 0 and i % 10 == 0:
                    s += "\n"
                s += o[i]
            print(s)
            print("\n")

        def same_obs(a, b):
            ao = ms._node[a][Graph._node_obs][0]
            bo = ms._node[b][Graph._node_obs][0]
            print(ao == bo)

        def find_p5(obs):
            a = np.where(np.array(obs, dtype=np.int8).reshape(10, -1) == 5)
            return [a[0][0], a[1][0]]

        error = 0

        for node in ms._crossing_nodes:
            obs = ms._node[node][Graph._node_obs][0]
            next_action = ms._node[node][Graph._node_action][0]
            next_nodes = ms._node[node][Graph._node_next]

            base_p5 = find_p5(obs)

            for na, nnd in zip(next_action, next_nodes):
                for nn in nnd:
                    if nn is not None:
                        next_p5 = find_p5(ms._node[nn][Graph._node_obs][0])
                    else:
                        break

                    if na == 0:
                        if next_p5[1] == base_p5[1] and (next_p5[0] == base_p5[0] or next_p5[0] == base_p5[0] - 1):
                            pass
                        else:
                            print(0)
                            print(f"{node} -{na}-> {nnd}")
                            error += 1
                    if na == 1:
                        if next_p5[1] == base_p5[1] and (next_p5[0] == base_p5[0] or next_p5[0] == base_p5[0] + 1):
                            pass
                        else:
                            print(1)
                            print(f"{node} -{na}-> {nnd}")
                            error += 1
                    if na == 2:
                        if next_p5[0] == base_p5[0] and (next_p5[1] == base_p5[1] or next_p5[1] == base_p5[1] - 1):
                            pass
                        else:
                            print(2)
                            print(f"{node} -{na}-> {nnd}")
                            error += 1
                    if na == 3:
                        if next_p5[0] == base_p5[0] and (next_p5[1] == base_p5[1] or next_p5[1] == base_p5[1] + 1):
                            pass
                        else:
                            print(3)
                            print(f"{node} -{na}-> {nnd}")
                            error += 1

        print(f"error: {error}")
        breakpoint()

    @staticmethod
    def sokoban_vi_test():
        graph = IO.read_disk_dump(f"{P.model_dir}Boxoban-Val-v1-optimal.pkl")
        from src.module.env.sokoban import Sokoban
        visited_graph = set()
        print({
            "^": 0,
            "v": 1,
            "<": 2,
            ">": 3,
        })

        def show(obs):
            s = ""
            for i in range(len(obs)):
                if i > 0 and i % 10 == 0:
                    s += "\n"
                s += obs[i]
            return s

        while True:
            env = Sokoban.make_env()
            last_obs = obs = env.reset()
            while True:
                action = env.action_space.sample()
                if last_obs in graph.obs_next:
                    for a in graph.obs_next[last_obs]:
                        next_obs = list(graph.obs_next[last_obs][a].keys())[0]
                        if next_obs not in visited_graph:
                            action = a
                            break
                obs, reward, done, info = env.step(action)

                if last_obs in graph.obs_next:
                    if action in graph.obs_next[last_obs]:
                        graph_obs = list(graph.obs_next[last_obs][action].keys())[0]
                        assert graph_obs == obs, f"next wrong:\n {show(last_obs)}\n -{action}-\n {show(obs)}\n\n {show(graph_obs)}"
                        visited_graph.add(last_obs)
                if obs in graph.obs_prev:
                    if action in graph.obs_prev[obs]:
                        if last_obs in graph.obs_prev[obs][action]:
                            visited_graph.add(last_obs)
                if obs in graph.obs_reward:
                    assert graph.obs_reward[obs] == reward, f"reward wrong: {graph.obs_reward[obs]} != {reward}"
                    # visited_graph.add(obs)
                
                last_obs = obs
                if done:
                    visited_percent = (len(visited_graph)) / (len(list(graph.obs_next.keys())))
                    print(f"{visited_percent:4.4f}")
                    break
        breakpoint()

    @staticmethod
    def multi_hash():
        from src.module.agent.memory.projector import RNNProjector
        import numpy as np

        a = np.random.rand(3,2,3)
        b = np.random.rand(3,2,3)
        c = np.random.rand(3,2,3)

        # print(f"{a}\n\n{b}\n\n{c}")

        last_obs = a
        obs = b
        projector = RNNProjector(id=0)
        last_obs, obs = projector.batch_project([last_obs, obs])
        print(f"first time:\n{last_obs}\n{obs}\n")
        last_obs = b
        obs = c
        last_obs, obs = projector.batch_project([last_obs, obs])
        print(f"second time:\n{last_obs}\n{obs}\n")

    @staticmethod
    def atari_graph_replay():
        graph = IO.read_disk_dump(f"{P.model_dir}StarGunner-optimal.pkl")
        from src.module.env.atari import Atari
        from src.module.agent.memory.projector import RNNProjector
        import time

        Logger.path = f"{P.log_dir}{P.env_name}-{Logger.get_date()}-test.log"

        projector = RNNProjector(0)
        P.env_name = P.env_name_list[int(24)]
        env = Atari.make_env()
        saved_traj = graph.general_info["max_total_reward_traj"]
        saved_last_obs = [e[0] for e in saved_traj]
        saved_action = [e[1] for e in saved_traj]
        # show info
        projector.reset()
        last_obs = obs = env.reset()
        last_obs, obs = projector.batch_project([last_obs, obs], first_frame=True)
        g_action = graph.obs_best_action[obs]
        ereward = 0
        current_obs = last_obs
        epi_step = 1
        while True:
            ereward += graph.obs_reward[current_obs]
            if current_obs in graph.obs_next:
                # if saved_last_obs[epi_step - 1] != current_obs:
                #     Logger.log("wrong.")
                #     breakpoint()
                snode = graph.obs_node[current_obs]
                snode_value = graph.node_value[snode]
                
                Logger.log(f"{saved_last_obs[epi_step - 1][-4:]} vs {current_obs[-4:]}; {saved_traj[epi_step - 1][3]} vs {graph.obs_reward[current_obs]}; {snode}:{snode_value}")
                e_action = saved_traj[epi_step - 1][1]
                epi_step += 1
            else:
                break
            current_obs = list(graph.obs_next[current_obs][e_action].keys())[0]
        saved_init_obs = graph.general_info["max_total_reward_init_obs"]
        Logger.log(f"save action replay: {last_obs[-4:]} vs {saved_init_obs[-4:]}; {ereward} vs {graph.general_info['max_total_reward']}; {epi_step} vs {len(saved_action)}")

    @staticmethod
    def atari_traj_replay():
        from src.module.env.atari import Atari
        from src.module.agent.memory.projector import RNNProjector

        Logger.path = f"{P.log_dir}{P.env_name}-{Logger.get_date()}-test.log"

        projector = RNNProjector(0)
        P.env_name = P.env_name_list[int(24)]
        env = Atari.make_env()
        # saved_traj = IO.read_disk_dump(f"{P.result_dir}221122100310-traj.pkl")
        saved_traj = IO.read_disk_dump(f"{P.result_dir}221122100635-traj.pkl")
        saved_last_obs = [e[0] for e in saved_traj]
        saved_action = [e[1] for e in saved_traj]
        saved_obs = [e[2] for e in saved_traj]
        saved_reward = [e[3] for e in saved_traj]
        saved_total_reward = sum(saved_reward)

        while True:
            # set env
            epi_step = 1
            projector.reset()
            last_obs = obs = env.reset()
            
            total_reward = 0
            while True:
                s_action = saved_action[epi_step - 1]
                action = s_action
                obs, reward, done, info = env.step(action)
                env.render(mode="rgb_array")

                plast_obs, pobs = projector.batch_project([last_obs, obs], epi_step == 1)
                if epi_step > 1:
                    assert saved_last_obs[epi_step - 2] == plast_obs, "wrong"
                total_reward += reward
                last_obs = obs
            
                if done:
                    Logger.log(f"saved reward: {saved_total_reward} total reward: {total_reward} steps: {epi_step}")
                    break
                epi_step += 1

    @staticmethod
    def atari_graph_reinteract():
        graph = IO.read_disk_dump(f"{P.model_dir}StarGunner-optimal.pkl")
        from src.module.env.atari import Atari
        from src.module.agent.memory.projector import RNNProjector
        import time

        Logger.path = f"{P.log_dir}{P.env_name}-{Logger.get_date()}-test.log"

        projector = RNNProjector(0)
        P.env_name = P.env_name_list[int(24)]
        env = Atari.make_env()
        epi_step = 1
        projector.reset()
        last_obs = obs = env.reset()
        plast_obs, pobs = projector.batch_project([last_obs, obs], epi_step == 1)

        saved_traj = graph.general_info["max_total_reward_traj"]
        saved_obs = [e[2] for e in saved_traj]
        saved_action = [e[1] for e in saved_traj]
            
        total_reward = 0
        while True:
            action = graph.obs_best_action[pobs]

            if action != saved_action[epi_step - 1]:
                Logger.log(f"#{epi_step} g-s A: {action}-{saved_action[epi_step - 1]}")
            obs, reward, done, info = env.step(action)
            epi_step += 1
            plast_obs, pobs = projector.batch_project([last_obs, obs], epi_step == 1)
            if pobs != saved_obs[epi_step - 2]:
                gnode = graph.obs_node[pobs]
                gnode_value = graph.node_value[gnode]
                Logger.log(f"{pobs[-4:]} - {saved_obs[epi_step - 2][-4:]} step {epi_step - 1} inconsistence, {gnode}:{gnode_value} - {reward}")
            env.render(mode="rgb_array")
            total_reward += reward
            last_obs = obs
        
            if done:
                Logger.log(f"total reward: {total_reward} steps: {epi_step}")
                break

    def read_graph(self):
        optimal_graph = IO.read_disk_dump(P.optimal_graph_path)
        breakpoint()

    def mujoco(self):
        import gym
        import d4rl

        # a = gym.make("halfcheetah-medium-expert-v2")
        # from src.module.env.mujoco import Mujoco
        # a = Mujoco.make_env(is_head=True)
        # breakpoint()
        # s = a.env.env.get_normalized_score(20)

        env = gym.make("halfcheetah-medium-expert-v2")

        dataset = d4rl.qlearning_dataset(env)
        # ['observations', 'actions', 'next_observations', 'rewards', 'terminals']
        # dataset["observations"].shape: (1998000, 17)
        # dataset["actions"].shape: (1998000, 6), array([-0.22293739, -0.7359478 , -0.8599511 ,  0.29579234, -0.8416547 , 0.43432042], dtype=float32)
        # dataset["next_observations"].shape: (1998000, 17)
        # dataset["rewards"].shape: (1998000,)
        # dataset["terminals"].shape: (1998000,)

        breakpoint()

    def load_gpt_model(self):
        import torch
        from src.util.dt.mingpt.model_atari import GPT, GPTConfig
        self.max_timestep = 3884
        self.model = GPT(GPTConfig(
            18, 
            90,
            n_layer=6, 
            n_head=8, 
            n_embd=128, 
            model_type="reward_conditioned", 
            max_timestep=self.max_timestep
        ))
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model.load_state_dict(torch.load("src/util/dt/model.pt"))
        self.model = self.model.module
        breakpoint()

    def named_tuple(self):
        import numpy as np
        obs_visited = {4: 44, 1: 11, 6: 66, 3: 33, 2: 22, 5: 55}
        max_obs_size = 5
        if len(obs_visited) > max_obs_size:
            s = sorted(obs_visited.items(), key=lambda i: i[1])    
            filterred_obs = np.array(s)[max_obs_size:, 0]   
            breakpoint()

    def football_env(self):
        from src.module.env.football import Football
        env = Football.make_env(False, True)


        # done = False
        # while not done:
        #     _, _, done, _ = env.step(0)
        # env.render()

        breakpoint()
        # obs = env.reset()
        # obs[0].shape

    def print_table_param(self):
        Funcs.print_obj([P.C, P])


if __name__ == "__main__":
    test = Test()
    # testable of content for testing:
    #
    # test.plot_maze()  # test graph gen. for maze env.
    # test.build_graph_test_manual()
    # test.build_graph_test_auto()
    # test.vi_test()
    # test.maze_vi_validation()
    # test.sokoban_vi_test()
    # test.multi_hash()
    # test.atari_graph_reinteract()
    # test.atari_graph_replay()
    # test.atari_traj_replay()
    # test.sokoban_vi_validation()
    # test.hashing_test()
    # test.make_atari_alternative_env()
    # test.atari_play()
    # test.nrnn()
    # test.rnn()
    # test.football()
    # test.test_pypullet()
    # test.test_sokoban()
    # test.test_tiny_sokoban()
    # test.read_highway_graph()
    # test.read_graph()
    # test.mujoco()
    # test.load_gpt_model()
    # test.named_tuple()
    test.football_env()
    # test.print_table_param()

# from ctypes import sizeof
# from src.util.imports.numpy import np
# from src.module.agent.memory.indexer import Indexer
# from multiprocessing import Pool, Process, Value, Queue, Lock, Manager
# from src.module.env.atari import Atari
# from src.util.tools import *
# from src.module.context import Profile as P
# from src.module.agent.memory.projector import RandomProjector

# import numpy as np
# from sklearn.neighbors import KDTree


# r = RandomProjector(0)
# a = np.random.random([1, 7084])

# b = r.batch_project([a])

# print(b)


# print([1, 2, 3, 4, 9, 4].index(0))





# -------------------------------------------------------------------
# import gym
# from gym import envs

# # print(envs.registry.all())

# gym.make("StarGunner-v0")

# -------------------------------------------------------------------

# from src.module.agent.memory.projector import CNNProjector
# import torch


# a = CNNProjector(0)
# b = torch.rand(2, 84 * 84)
# c = a.batch_project(b)
# print(c)

# -------------------------------------------------------------------

# import sys
# from src.util.tools import Funcs
# import ctypes

# class Storeage(dict):
#     def __init__(self) -> None:
#         self.ggg = None


# class Game:
#     def __init__(self) -> None:
#         pass


# a = dict()
# b = Storeage()
# c = list()
# d = list()


# print(sys.getsizeof(a), "B")
# print(sys.getsizeof(b), "B")
# print(sys.getsizeof(c), "B")
# print(sys.getsizeof(d), "B")

# a = dict()
# s = Funcs.matrix_hashing([1, 2, 3, 4, 5, 6, 7])
# a[s] = [90000, 5698]
# c = id(a[s])
# a[s] = [90000, 6698]

# print(sys.getsizeof(c), "B")
# print(sys.getsizeof(s), "B")

# print(a[s])
# value = ctypes.cast(c, ctypes.py_object).value
# print(value)

# a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# step = 3

# b = a[:step]
# print(b)
# c = [a[step]]
# print(c)
# d = a[step + 1:]
# print(d)

# -------------------------------------------------------------------
# another loop detection algorithm 

# import numpy as np
# import argparse
# import logging
# logger = logging.getLogger(__name__)


# def strong_connection(G, s, v, i, node_index, visited_stack):
#     """Tarjan's strong connection detection algorithm.
#     Args:
#         G (list): Directed graph (edge_list) to inspect.
#         s (int): Root node's index. Root means the start node of depth-first search.
#         v (int): Index of current node.
#         i (int): Counter of depth-first search.
#         node_index (np.array(np.int32)): Index list of nodes. Index of unvisited node is -1.
#         visited_stack (list): Use as stack of visited nodes.
#     Returns:
#         list: List of node list for each cycle found by this search.
#     Notes:
#         This function will call recursively; construct a depth-first search.
#     """
#     node_index[v] = i
#     i += 1
#     visited_stack.append(v)
#     logger.debug("v, G[v], visited_stack: {}, {}, {}".format(
#         node_index[v], G[v], visited_stack))

#     cycles = []
#     for w in G[v][:]:
#         logger.debug("\t{} to {}  node_index: {}".format(v, w, node_index))
#         if w < s:
#             G[v].pop(G[v].index(w))
#         elif w == s and w in visited_stack:
#             logger.debug("cycle found!: {}".format(visited_stack))
#             cycles.append(visited_stack)
#         elif node_index[w] == -1:
#             cycle = strong_connection(
#                 G, s, w, i, np.copy(node_index), visited_stack[:])
#             if len(cycle) > 0:
#                 cycles.append(cycle)
#     return cycles


# def find_cycles_in_graph_tarjan(G):
#     """Enumerate cycles by Tarjan's strong connection detection algorithm.
#     Args:
#         G (list): Directed graph (edge_list) to inspect.
#     Returns:
#         list: List of node list for each cycle.
#     """
#     node_index = np.zeros(len(G), dtype=np.int32)
#     node_index[:] = -1
#     cycles = []
#     for v in range(len(G)):
#         print(
#             "{}: --------------------------------------------------".format(v))
#         visited_stack = []
#         cycles_v = strong_connection(
#             G, v, v, 1, np.copy(node_index), visited_stack)
#         if len(cycles_v) > 0:
#             print(cycles)
#             cycles.extend(cycles_v)
#     return cycles


# def main():
#     # https://stackoverflow.com/questions/25898100/enumerating-cycles-in-a-graph-using-tarjans-algorithm
#     G = [[1], [4, 6, 7], [4, 6, 7], [4, 6, 7],
#          [2, 3], [2, 3], [5, 8], [5, 8], [], []]
#     # https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm#/media/File:Tarjan%27s_Algorithm_Animation.gif
#     # G = [[1], [2], [0], [1, 2, 5], [2, 6], [3, 4], [4], [5, 6]]
#     # G = [[1], [2, 3], [0], [1], [0]]
#     G = [[1], [2, 3], [0], [4], [0]]
#     # G = [[190, 138, 161, 168, 215, 253], [110, 192, 212, 226, 282, 67], [51, 61, 90, 93, 157, 226, 60], [212, 299, 237], [201, 241, 34, 45, 115, 134, 141, 156, 177, 199, 220, 234, 259], [120, 136, 147, 263, 9, 18, 51, 59, 91, 102, 125, 142, 157, 165, 185, 186, 201, 210, 267, 280], [276], [25, 262], [110, 282, 139, 151], [100, 223, 277], [5, 109, 198, 51, 160], [60, 95, 160, 169, 254, 69, 207], [269, 294], [5, 164, 198, 125], [0, 114, 160, 190, 161, 215], [5, 142, 185], [29, 53, 125, 203, 42, 94, 117, 191, 205], [4, 234], [17, 124, 149, 175, 64], [76, 98], [10, 124, 136, 160], [25, 168, 192], [19, 199, 65], [51, 93, 184, 276, 138, 233], [2, 6, 95, 178], [7, 14, 15, 17, 42, 62, 86, 153, 158, 159, 172, 183, 191, 192, 193, 194, 212, 225, 248, 256, 262, 273, 287], [105, 192, 198], [209, 84, 100], [109, 161, 198, 289], [73, 135, 35, 42, 45, 46, 57, 66, 78, 79, 82, 94, 103, 117, 128, 135, 156, 171, 173, 199, 205, 209, 286, 297], [51, 61, 93, 110, 128, 226, 283], [23, 106, 160, 276], [8, 10, 212, 276, 138, 149, 206, 215, 236], [4, 234], [61], [103, 257, 276, 219], [109, 105], [136], [95, 99, 156, 178, 43, 218], [105, 280, 109], [142, 173, 201, 187], [18, 5, 120, 136, 153, 293], [95, 54, 204], [192, 292], [29, 105, 232, 283, 128, 173, 209], [4, 53, 143, 152, 203, 115, 156, 220, 259], [61, 71, 201, 214, 254], [160, 169, 190, 268], [45, 4, 29, 45, 143, 233], [232], [25, 139, 193], [8, 44, 32, 192, 276, 52, 60, 114, 124, 151, 233, 283], [27, 39, 53, 165, 207, 241, 283, 291, 293], [12, 21, 24, 27, 39, 259, 77, 88, 111, 115, 119, 132, 134, 143, 178, 181, 195, 205, 220, 244, 259, 291, 293, 296], [160, 169, 221, 247], [201, 241, 145], [42, 25, 105, 141, 82, 86, 94, 194, 256, 297], [54, 140, 234, 122, 200, 217, 222, 235], [179, 279], [35, 53, 125, 203, 88], [26, 95, 178, 254, 63, 69, 197, 207], [60, 53, 62, 71, 105, 234, 119, 124, 164, 190, 254, 283], [201, 234, 282, 119, 143, 163, 178, 284], [25, 56, 105, 232, 294, 94, 194], [132, 126, 249], [27, 52, 53, 105, 203, 232, 283], [115, 175], [89, 164, 71, 127, 157], [12, 165, 185, 201, 273, 81], [5, 15, 136, 142], [16, 106, 160], [62, 234, 119, 254, 265, 284], [3, 127, 257, 276, 224, 240], [231, 286], [25, 183], [25, 101, 124, 131, 78, 191, 248], [190, 192, 299], [234, 257, 260, 276, 85], [182, 213], [121, 156, 169, 183, 193, 199, 262, 150, 170], [54, 42, 57], [151, 190, 144], [109, 214, 108], [25, 273], [0, 23, 32, 160, 276, 138], [25, 252, 169, 212], [107], [47, 158, 257, 276], [5, 136, 153, 227, 267], [71, 72, 164, 198], [60, 61, 157, 192, 201, 164], [57, 29, 237], [124, 175, 263, 229], [5, 52, 60, 0, 114, 114, 149, 168, 233, 283], [233, 234, 280, 99], [6, 26, 43, 69, 172, 100, 104, 146, 170, 197, 204, 218], [5, 136, 102], [67, 164, 198, 214, 127], [5, 136, 210], [26, 43, 65, 27, 95, 100, 104, 170, 218], [147, 211, 277], [15, 25, 124, 131, 232, 294, 248], [53, 62, 105, 234, 178, 214], [248, 257, 276, 219, 250, 278], [52, 51, 93, 106, 184, 257], [27, 35, 42, 46, 78, 82, 86, 94, 95, 103, 25, 109, 115, 119, 143, 173, 178, 194, 198, 214, 220, 256, 266, 287, 296], [16, 23, 52, 8, 51, 110, 192, 212, 226, 257, 114, 151], [42, 46, 16, 25, 29, 56, 105, 232, 245], [37, 55, 120, 175, 174, 264], [30, 51, 53, 56, 75, 101, 105, 108, 110, 203, 227, 242, 271, 289], [44, 67, 212, 257, 114, 139, 151, 190, 239, 283], [87], [110, 109], [124, 175, 162], [32, 192, 276, 149, 161, 168, 206, 215], [11, 32, 175], [14, 78, 29, 75, 105, 232, 243], [68, 129], [24, 34, 53, 142], [28, 31, 116, 130, 202, 288], [18, 37, 91, 92, 113, 136, 166, 228, 264], [49, 169, 183, 193, 199, 262, 150], [97, 136, 148], [79, 29], [7, 9, 15, 41, 55, 64, 25, 131, 232, 160, 162, 165, 174, 184, 226, 229, 248, 257, 258, 280, 290], [35, 66, 88, 117, 29, 135, 213, 132, 205, 297], [109, 242], [98, 19, 240, 246], [114, 51, 93, 106, 110, 257, 283], [35, 29, 59, 105, 125, 203], [9, 5, 124, 136, 247], [7, 15, 40, 55, 105, 175, 248, 266], [126, 295], [131], [36, 50, 169], [66, 77, 112, 286, 297], [9, 18, 59, 91, 102, 113, 133, 142, 148, 160, 165, 184, 210, 228, 255, 257, 264, 267, 280], [201, 219, 234, 275], [89, 96, 147, 186, 196], [128, 29, 44, 193], [48, 152, 154, 200, 217, 238], [29, 171, 256], [24, 39, 52, 53, 283, 187, 291], [34, 45, 115, 134, 4, 118, 201, 156, 220, 259], [13, 261], [55, 108, 124, 131, 175, 232, 174], [108, 120, 136, 258, 264], [59, 89, 91, 97, 122, 186, 211, 223, 255], [263], [64, 25, 150, 153], [41, 124, 175, 153], [58, 73, 144, 267, 154], [134, 4, 53, 143, 259], [18, 59, 5, 136, 147, 186, 267], [3, 72, 257, 276, 224], [14, 25, 242, 158], [26, 65, 150, 22, 99, 170, 218], [60, 192, 201, 212, 215, 226, 164, 179], [28, 47, 119, 235, 257, 276], [116, 119, 210, 272, 176, 202], [16, 23, 26, 69, 124, 138, 79, 95, 99, 156, 169, 183, 199, 262, 161, 170, 190, 207, 215, 247, 268], [70, 198, 289], [201, 282, 221, 265], [11, 115], [71, 72, 125, 127, 155, 198, 227], [12, 22, 81, 132, 53, 259, 273, 195, 244, 293], [175, 216, 228], [188, 208], [93, 175, 192], [26, 36, 49, 50, 69, 74, 150, 167, 170, 207, 247, 268, 285], [90], [282], [68, 146, 117], [201, 241, 187, 251], [4, 29, 199], [33, 37, 40, 41, 55, 64, 92, 93, 140, 162, 169, 174, 167, 188, 216, 226, 229, 258, 266, 290], [94, 117, 16, 29, 125, 203, 268], [122, 57, 179, 254], [6, 26, 43, 104, 95, 99, 197, 218], [63, 122, 154, 57, 140, 234, 200, 207, 222, 235, 279], [103, 29, 105, 186], [124, 136, 257], [95, 105, 187, 280], [26, 49, 150, 166, 167, 170, 269, 274, 285], [52, 149, 32, 93, 114, 276, 233], [81, 201, 276, 298], [103, 29, 44, 283, 209], [95, 46, 102, 105, 214], [53, 109, 198, 214], [5, 236, 201], [1, 20, 58, 144, 161, 151, 234, 253, 268, 270, 272, 299], [75, 101, 109], [8, 44, 67, 151, 164, 179, 198, 206, 260, 270, 292, 299], [49, 150, 169, 183, 199, 220, 262, 292, 285], [154, 140, 151, 179, 234], [43, 38, 95, 99, 178], [12, 195, 53, 68, 165, 273], [109, 198, 214, 265, 203], [30, 51, 53, 56, 70, 72, 75, 96, 123, 125, 127, 155, 204, 203, 227, 243, 289], [19, 49, 150, 170, 183, 220, 269, 285], [113, 120, 136], [22, 34, 38, 80, 81, 111, 141, 145, 163, 164, 177, 179, 187, 189, 221, 224, 251, 254, 265, 275, 298], [32, 212, 257, 276, 236], [21, 27, 35, 39, 82, 88, 94, 117, 29, 56, 125, 135, 205, 259, 296, 297], [123, 281], [124, 175, 258], [165, 184, 124, 136], [195, 53, 165, 196, 273, 293], [106], [84, 78, 213], [116, 130, 119, 288], [157, 5, 67], [10, 67, 151, 179, 190, 236, 237, 239, 272], [66, 77, 53, 135, 259], [53, 56, 75, 108, 127, 203, 109, 191, 198, 227], [179, 71, 162, 201, 265], [17, 18, 25], [198, 243], [0], [121, 234, 257, 275], [166, 167, 183, 269, 285], [1, 190, 247], [190, 61, 110, 160], [39, 52, 53, 142, 203, 283], [196, 138, 289], [25], [5, 44, 60, 67, 151, 179, 51, 110, 192, 257, 276, 283], [96, 186, 138, 198, 267, 290], [140, 175, 270], [5, 93, 226], [92, 174, 108, 124, 145, 175, 258], [83], [7, 15, 27, 40, 42, 46, 55, 78, 173, 194, 131, 175, 296], [45, 62, 99, 25], [4, 20, 38, 58, 85, 99, 119, 121, 143, 154, 163, 178, 200, 217, 64, 249, 252, 275, 284], [28, 202, 119, 159, 257], [201, 5, 124, 136, 280], [57, 135, 29], [53, 105, 203, 232, 296], [215, 0, 14, 32, 114, 160], [172, 25, 253], [22, 141, 145, 165, 201, 283, 251, 293], [158, 191, 16, 25, 75], [46, 78, 29, 105, 107, 232], [165, 5, 124, 136, 206], [42, 94, 16, 29, 56, 63, 105, 176, 203], [222, 57, 179, 296], [9, 124, 175, 227, 290], [176, 159, 278], [132, 53, 125, 165], [48, 140], [63, 60, 179, 254], [212, 226, 124, 175], [14, 172, 25, 116, 155], [2, 63, 69, 122, 197, 207, 24], [120, 277], [169, 85, 175], [3, 13, 28, 44, 47, 52, 80, 85, 114, 121, 151, 202, 219, 236, 239, 240, 246, 250, 144, 278], [174, 255, 136, 147, 264], [77, 180, 181, 244], [85, 250, 103, 257, 276], [190, 192, 270], [49, 50, 150, 167, 170, 134, 169, 285], [91, 92, 223, 229, 9, 147], [30, 109, 198], [56, 203, 227, 109, 164, 198, 214], [56, 109, 198, 214, 265], [73, 152, 140], [82, 86, 117, 25, 56, 105], [230, 294], [33, 140, 175], [220, 4, 45, 53, 105, 143], [116, 246, 127, 257], [12, 81, 111, 195, 244, 53, 201], [124, 51, 61, 160], [115, 156, 4, 45, 53, 105, 143], [3, 8, 10, 23, 28, 44, 47, 80, 85, 138, 149, 206, 219, 233, 236, 240, 250, 252, 201, 257, 278, 298], [120, 147], [119, 210, 288], [241], [95, 99, 109, 143, 53, 62, 105, 234], [118, 290], [38, 67, 139, 163, 221, 201, 234], [22, 27, 39, 173, 209, 224, 72, 154, 201, 291, 293], [207, 11, 60, 160, 169, 179, 254], [156, 4, 29, 45, 143, 275], [29], [21, 53, 203], [137], [186, 196, 5, 138, 147, 153, 227], [118, 245], [26, 60, 95, 99, 156, 160, 169, 178, 183], [167, 285, 169, 183, 220, 262], [18, 92, 120, 175, 230, 263], [15, 82, 194, 29, 56, 105, 203, 268], [177, 4, 201], [222, 235, 57, 179], [20, 190, 234], [159, 25], [237, 252, 234, 276]]
#     print(find_cycles_in_graph_tarjan(G))


# if __name__ == '__main__':
#     # args
#     parser = argparse.ArgumentParser(
#         description="Tarjan's strong connection detection algorithm.")
#     parser.add_argument(
#         "-d", "--debug", action="store_true", help="Debug mode.")
#     args = parser.parse_args()

#     # logger
#     if args.debug:
#         LOG_LEVEL = logging.DEBUG
#     else:
#         LOG_LEVEL = logging.INFO
#     formatter = logging.Formatter(
#         fmt="[%(asctime)s] %(levelname)s [%(name)s/%(funcName)s() at line %(lineno)d]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
#     logger.setLevel(LOG_LEVEL)
#     # stdout
#     stream_handler = logging.StreamHandler()
#     stream_handler.setFormatter(formatter)
#     stream_handler.setLevel(LOG_LEVEL)
#     logger.addHandler(stream_handler)
#     main()





# -------------------------------------------------------------------
# this section test the hadmard product

# import torch
# import numpy as np

# p = np.array([
#     [0, 1, 0, 0],
#     [0, 0, 0, 1],
#     [1, 0, 0, 1],
#     [1, 0, 0, 1],
# ])

# a = torch.from_numpy(p)

# b = torch.rand(4)

# c = a * b

# print(c)



# -------------------------------------------------------------------

# a = Memory()

# a['a'] = [0, 100]
# a['b'] = [0, 200]


# if 'c' not in a:
#     a['c'] = [1, 300]

# print('for')
# for key in a:
#     print(key)

# print(f"a[\'a\'] {a['a']}")

# print(len(a))

# print(a.max_value)
# print(a.max_value_init_obs)
# print(dict(a))
# print(a)




# -------------------------------------------------------------------

# a = dict()
# sample = set()
# env = Atari.make_env(False)
# for _ in range(5000):
#     obs = env.reset()  # random_noops = 15
#     # obs = env.reset()  # random_noops = 30
#     # obs = env.reset()  # random_noops = 11
#     for i in range(1):
#         obs, reward, done, info = env.step(i)
#     # h = Indexer.get_ind(obs)
#     h = Funcs.matrix_hashing(obs)
#     if h not in a:
#         a[h] = 1
#     else:
#         a[h] += 1

# # print(a)
# print(len(a))

# -------------------------------------------------------------------

# o = IO.read_disk_dump(P.optimal_graph_path)
# m = max([i[1] for i in o.values()])
# print(f'max value: {m}')
# print(f'o len: {len(o)}')


# one = set()
# for key in o:
#     if o[key][1] == m:
#         one.add(key)
# print(f'num max point {len(one)}')

# two = set()
# for key in o:
#     if o[key][1] == m:
#         two.add(o[key][2])

# two_out = set()
# two_end = set()
# for e in two:
#     if e in o:
#         if o[e][1] != m:
#             two_out.add(e)
#     else:
#         two_end.add(e)
# print(f'two_out {len(two_out)}')
# print(f'two_end {len(two_end)}')



# xx = []
# for e in ml:
#     if o[e][1] != m:
#         xx.append(e)
# print(f'num ending obs from last max value point: {len(xx)}')

# m2 = set()
# for key in o:
#     if o[key][1] == m:
#         m2.add(key)

# for e in m2:
#     if o[e][1] == m:
#         if e not in o:
#             print(f'pure ending point with max value: {x}')


# env = Atari.make_env(False)
# random_projector = RandomProjector(0)
# obs = env.reset()
# total_reward = 0
# while True:
#     obs = random_projector.batch_project([obs])[0]
#     obs = Indexer.get_ind(obs)
#     a = o[obs]
#     obs, reward, done, info = env.step(0)

#     total_reward += reward

#     if done:
#         print(f'R: {total_reward}')
#         break
# print('fin')



# -------------------------------------------------------------------
# a = Queue()
# a.put(1)
# a.put(2)
# a.put(3)
# print(a.qsize())


# print(a.get())
# print(a.get())
# print(a.get())
# -------------------------------------------------------------------

# import gym 

# from gym import envs

# a = [env_spec.id for env_spec in envs.registry.all()]

# for b in a :
#     print(b)

# -------------------------------------------------------------------

# vram size test

# import torch
# import time

# big_num = 1000000000
# small_num = 24 * int(1024 * 1024 / 4)
# times = 5

# print(big_num * 4 / 1024 / 1024)
# print(small_num * 4 / 1024 / 1024)


# class M(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.op = torch.rand(big_num * 2).to("cuda:0")

#     def forward(self, input):
#         return self.op


# def a():
#     print("a")
#     x = torch.rand(big_num).to("cuda:0")
#     print(x.shape)
#     time.sleep(times)


# def b():
#     print("b")
#     m = M()
#     y = m(None)
#     print(y.shape)
#     time.sleep(times)


# def c():
#     print("c")
#     x = torch.rand(small_num).to("cuda:0")
#     print(x.shape)
#     time.sleep(times)


# # a()
# b()
# # c()
# -------------------------------------------------------------------

# to test the dag algorithm
# from src.module.agent.memory.iterator import Iterator
# from src.util.imports.numpy import np

# import networkx as nx
# import matplotlib.pyplot as plt

# np.random.seed(6)


# def show_graph(adj, id):
#     g = nx.DiGraph()
#     rows, cols = np.where(adj == 1)
#     g.add_edges_from(zip(rows.tolist(), cols.tolist()))
#     pos = nx.random_layout(g)
#     nx.draw_networkx(g, pos)
#     # nx.draw(g, node_size=100, with_labels=False)
#     plt.savefig(f'output/{id}.png', format='png')
#     plt.close()


# adj = [
#     [0, 1, 0, 0],
#     [0, 0, 1, 0],
#     [1, 0, 0, 1],
#     [0, 0, 0, 0],
# ]
# adj = [
#     [0, 1, 0, 0],
#     [0, 0, 0, 0],
#     [0, 0, 0, 1],
#     [0, 0, 0, 0],
# ]
# adj = np.array(adj, dtype=np.float32)
# # adj = np.random.rand(6, 6)
# # adj = np.where(adj > 0.8, 1.0, 0.0)
# print(adj)
# show_graph(adj, 0)

# iterator = Iterator(0)
# etm = iterator.build_dag(adj)
# print(etm)

# adj = adj - etm
# print(adj)
# show_graph(adj, 1)


# -------------------------------------------------------------------

# from src.module.env.simple_scene import SimpleScene

# env = SimpleScene.make_env()

# print(env.reset())
# for _ in range(10):
#     obs, _, done, _ = env.step(0)
#     if done:
#         print(obs)
#         break

# -------------------------------------------------------------------
# 

# from src.module.env. import Maze
# env = Maze.make_env()
# breakpoint()