import gym
import cv2
from double_dqn import DQN
from double_dqn import Experience_replay
import numpy as np
import time
import csv

#used to delay epsilon
EPSILON_DECAY = 300000
FINAL_EPSILON = 0.1
INITIAL_EPSILON = 1.0
#used to set the threshold of experience replay
EPPERIENCE_REPLAY_THRESHOLD = 5000
#extract last 3 frams in games in order to consider enemy's bullets early.
NUM_FRAMES = 3




#train function ise used to receive inputs of<the total number of frams in training, two neural network,
#env of OpenAI GYM, experience_buffer, and buffer used to saved model>
def DQN_train(num_frames,deep_q,env,experience_replay,input_buffer):
    #receive 3 frames and convert them into MDP, curr_state is used to predict action by neural network
    curr_state = convert_process_buffer(input_buffer)
    #initial variables
    epsilon=1
    total_reward = 0
    #record score of each eposido
    score=[]
    csvfile=open("score.csv","w")
    writer = csv.writer(csvfile)
    writer.writerow(["score"])
    #num_frames is used to set the time of training, we use the number frames as counter
    observation_num = 0
    for observation_num in range(num_frames):
        # Slowly decay the learning rate
        if epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON-FINAL_EPSILON)/EPSILON_DECAY
        #initial state is used to calcuate q valule by q-network
        initial_state = convert_process_buffer(input_buffer)
        input_buffer = []
        #predict movement based on epsilon_greedy
        predict_movement, predict_q_value = deep_q.epsilon_greedy(curr_state, epsilon)
        #The sum of reward is the reward after action. 
        reward, over = 0, False
        for i in range(NUM_FRAMES):
            temp_observation, temp_reward, temp_done, _ = env.step(predict_movement)
            reward += temp_reward
            input_buffer.append(temp_observation)
            #check whether game over
            over = over | temp_done

        #if game over, print message and store data
        if over:
            print("The observation_num is: ",observation_num)
            print("total reward of this eposido is:  ", total_reward)
            print("The score list:", score)
            score.append(total_reward)
            env.reset()
            total_reward = 0
            #print(self.score)
        #get new state of 3 frames 
        new_state = convert_process_buffer(input_buffer)
        #save samples inexperience replay until size higher than threshold
        experience_replay.add(initial_state, predict_movement, reward, over, new_state)
        total_reward += reward
        env.render()
        #time.sleep(0.1)
        if experience_replay.size() > EPPERIENCE_REPLAY_THRESHOLD:
            s_batch, a_batch, r_batch, d_batch, s2_batch = experience_replay.sample(16)
            deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch, observation_num)
            #deep_q.target_train()

        # Save the network every 100000 iterations
        if observation_num % 10000 == 9999:
            print("Saving Network")
            deep_q.save_network("saved.h5")
            for word in score:
                writer.writerow([word])

        observation_num += 1


#convert_process_buffer used to compress the image from Atari game, because the outputs of OpenAI GYM in Atari game are images, 
#each iamge is one frame, each frame incluing states, rewards and other information of game, we need compress it 
#and extract the elements which can be express by MDP
def convert_process_buffer(input_buffer):
    #Converts the list of NUM_FRAMES images in the process buffer
    #into one training sample
    #convert original image from Space Invaders to the gray image with size of 84*94, in order to compress input data
    gray_image = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 90)) for x in input_buffer]

    #the self.process_buffer contains the last 3 full sized 192*160*3 pictures.
    states_frame = [x[1:85, :, np.newaxis] for x in gray_image]

    output_buffer=np.concatenate(states_frame,axis=2)
    #concatenate used to combine all array together
    return output_buffer


def main():
    env = gym.make('SpaceInvaders-v0')
    env.reset()
    experience_replay = Experience_replay(100000)

    # Construct appropriate network based on flags
    dqn = DQN()
    #deep_q.load_network('saved.h5')
    # A buffer that keeps the last 3 images
    input_buffer = []
    # Initialize buffer with the first frame
    s1, r1, d1, other1 = env.step(env.action_space.sample())
    s2, r2, d2, other2 = env.step(env.action_space.sample())
    s3, r3, d3, other3 = env.step(env.action_space.sample())
    input_buffer = [s1, s2, s3]
    env.render()
    score=[]

    #the first number is used to set training times. 
    DQN_train(20000,dqn,env,experience_replay,input_buffer)


if __name__ == '__main__':
    main()