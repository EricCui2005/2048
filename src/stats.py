from scipy.stats import norm
import math

def compare_agents(success1, trials1, success2, trials2, alpha=0.05):
    # Calculate proportions
    p1 = success1 / trials1
    p2 = success2 / trials2
    
    # Calculate pooled proportion
    pooled_p = (success1 + success2) / (trials1 + trials2)
    
    # Calculate standard error
    se = math.sqrt(pooled_p * (1 - pooled_p) * (1/trials1 + 1/trials2))
    
    # Calculate z-score
    z = (p1 - p2) / se
    
    # Calculate p-value (one-tailed test)
    p_value = 1 - norm.cdf(z)
    
    # Print results
    print(f"Agent 1 Win Rate: {p1:.3f} ({success1}/{trials1})")
    print(f"Agent 2 Win Rate: {p2:.3f} ({success2}/{trials2})")
    print(f"Z-score: {z:.3f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Statistically significant: {p_value < alpha}")
    
    return z, p_value
    
success_random = 0.0 * 10000
trials_random = 10000
success_small = 0.0  * 10000
trials_small = 10000
success_large = 0.0  * 10000
trials_large = 10000

print('small vs random')
z_score, p_value = compare_agents(success_small, trials_small, success_random, trials_random)
print('large vs random')
z_score, p_value = compare_agents(success_large, trials_large, success_random, trials_random)
print('small vs large')
z_score, p_value = compare_agents(success_large, trials_large, success_small, trials_small)
