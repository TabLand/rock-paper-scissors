import numpy as np
total = 0.0
steps = [1,2,-1,-2]
pick  = np.random.randint(0,len(steps))
step  = steps[pick]

def sequence(observation, configuration):
    global total, step, steps
    total = total + step
    print("Sequence Bot - My step is " + str(step))
    print("Sequence Bot - My total is " + str(total))

    if np.random.randint(50) == 1:
        #every so often, change speed unpredictably
        pick  = np.random.randint(0,len(steps))
        step  = steps[pick]

    result = int(np.floor(total % 3).item())

    return result

agents = {
    "sequence" : sequence
}
