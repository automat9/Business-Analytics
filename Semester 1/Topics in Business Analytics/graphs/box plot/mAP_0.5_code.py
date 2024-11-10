# Import matplotlib
import matplotlib.pyplot as plt

# Extract mAP_0.5 values for each training session 
mAP_first = [0.445690, 0.679142, 0.687740, 0.697310, 0.708650]
mAP_second = [0.573300, 0.639870, 0.652820, 0.669250, 0.717520]
mAP_third = [0.414000, 0.651232, 0.667870, 0.689890, 0.716640]

# Plot config
plt.figure(figsize=(8, 6))
plt.boxplot([mAP_first, mAP_second, mAP_third], labels=["First", "Second", "Third"])
plt.ylabel("mAP_0.5")
plt.grid(True, alpha=1)
plt.title("Box Plot of mAP_0.5 Scores by Training Session")

# Show the plot
plt.show()
