# Import matplotlib
import matplotlib.pyplot as plt

# Extract mAP_0.5:0.95 values for each training session 
mAP_first = [0.20744, 0.361355, 0.390985, 0.39972, 0.41304]
mAP_second = [0.31796, 0.3581, 0.36494, 0.37245, 0.40601]
mAP_third = [0.19421, 0.374265, 00.38932, 0.39776, 0.41816]

# Plot config
plt.figure(figsize=(8, 6))
plt.boxplot([mAP_first, mAP_second, mAP_third], labels=["First", "Second", "Third"])
plt.ylabel("mAP")
plt.grid(True, alpha=1)
plt.title("Box Plot of mAP_0.5:0.95 Scores by Training Session")

# Show the plot
plt.show()
