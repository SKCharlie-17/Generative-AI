import numpy as np
import matplotlib.pyplot as plt
def softmax_with_temperature(logits, temperature):
    logits = np.array(logits)  # Convert input list to numpy array
    exp_logits = np.exp(logits / temperature) # Scale logits by temperature and exponentiate
    return exp_logits / np.sum(exp_logits)  # Normalize to get probabilities
# Raw scores for 5 tokens
logits = [2.0, 1.0, 0.1, -1.0, -2.0]
temperatures = [0.2, 0.5, 1.0, 2.0]
# Calculate probabilities for each temperature
probabilities = {temp: softmax_with_temperature(logits, temp) for temp in temperatures}
# Plotting
for temp, probs in probabilities.items():
    plt.plot(range(1, len(probs) + 1), probs, label=f'Temperature = {temp}')

plt.title("Effect of Temperature on Softmax Probability Distribution")
plt.xlabel("Token Index")
plt.ylabel("Probability")
plt.legend()
plt.grid()
plt.show()



# #What is Temperature in LLMs?
# Temperature is a parameter that determines the probability distribution of the next word in text generation. It controlling the randomness and divesity of the output:

# ###Low Temperature (e.g., 0.0–0.4):
#  Results are deterministic, precise, and repetitive.
# ###Moderate Temperature (e.g., 0.5–1.0):
#  Outputs become more diverse while maintaining coherence.
# ###High Temperature (e.g., 1.1–2.0):
#  Text becomes creative and experimental, often at the cost of relevance and focus.


# What The Numbers Mean
# Let’s look at the probabilities for different temperatures:

# ##Temperature = 0.2 (Very Cold)
# Token 1: 99.3%

# Token 2: 0.67%

# Other tokens: nearly 0%

# This makes the model very focused on the highest-scoring token.
# ##Temperature = 2.0 (Very Hot)
# Token 1: 42.5%

# Token 2: 25.8%

# Token 3: 16.4%

# Token 4: 9.5%

# Token 5: 5.8%

#  This spreads out the probabilities, making the model more random.
