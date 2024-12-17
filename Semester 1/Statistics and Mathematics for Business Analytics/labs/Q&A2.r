#1)	How is a population defined in statistical analysis?
The complete set of elements or observations of interest that share a specific characteristic or set of characteristics.

#2)	What does a population parameter refer to?
A numerical value that represents a specific characteristic of the entire population, such as the mean or standard deviation.

#3)	What are sample statistics in a statistical context?
Numerical measures (e.g., sample mean, sample variance) calculated from a subset of the population to make inferences about the population parameters.

#4)	What does ordinal-scaled data mean?
Data that can be ordered or ranked, but the differences between the ranks do not have meaningful or equal intervals.

#5)	How is cross-sectional data characterized?
Data collected at a single point in time or over a short period, capturing information from multiple subjects or entities at once.

#6)	What is the main purpose of inference in statistics?
To use sample data to estimate unknown population parameters and make inferences about the entire population.

#7)	Which statement regarding outliers is accurate?
Outliers can indicate measurement errors or true variability in the data.

#8)	What defines an outlier in statistical terms?
A value that lies significantly outside the overall pattern of data points.

#9)	How do outliers influence the mean of a dataset?
They can skew the mean significantly higher or lower.

#10)	If a bag contains 3 red balls, 5 blue balls, and 2 green balls, what is the probability of drawing a red ball?
3/10

#11)	In an experiment where a coin is flipped 1,000 times, resulting in 490 heads, what probability of landing heads is estimated using the relative frequency method?
0.490

#12)	What statement best describes the significance of random sampling in statistical research?
Random sampling eliminates bias by ensuring everyone in the population has an equal chance of selection, enhancing the generalizability of findings.

#13)	In a normal distribution, what percentage of data lies within one standard deviation of the mean?
About 68%

#14)	In a normal distribution, approximately what percentage of data falls within two standard deviations of the mean?
About 95%

#15)	What does the mode represent in a dataset?
The value that occurs most frequently in a dataset.
 
 

#A toy factory produces electronic toys, and historical data shows that 8% of these toys are defective. If a random sample of 200 toys is taken from the production line, calculate the following:  
#a)	What is the probability of finding exactly 12 defective items in this sample?  
#b)	What is the probability of finding at most 10 defective items in this sample?  
#c)	Discuss how the probabilities calculated in parts (a) and (b) would change if the sampling size were increased to 300 components. What implications does this have for quality control in the factory? 
#d)	Implications for quality control 
 
a <- B~(200,0.08)
b <- P(X=12) = 0.06532 ANSWER CORRECT, LIVIO WRONG
c <- P(X ≤10) or P(X<11) = 0.06912 
d <- Although the overall number of defects would increase as 8% of 300 is more than 8% of 200,
the distribution becomes more centred around its new mean. In other words,
the chances of finding exactly 12 or at most 10 is significantly lowered because there’s now a higher chance of finding more defects.


#A coffee shop receives an average of 5 customers every hour.
#a) What is the probability that the coffee shop receives exactly 4 customers in a given hour?
Using the Poisson distribution formula, we have:
• Average number of customers per hour (λ) = 5
• Number of customers (k) = 4
The formula is:
P(X = k) = (e^(-λ) * λ^k) / k!
Calculating this:
P(X = 4) = (e^(-5) * 5^4) / 4!
= (0.0067 * 625) / 24
≈ 0.1755 (or about 17.55%).


#b) What is the probability that the coffee shop receives no more than 3 customers in a given hour?

To find the probability of receiving no more than 3 customers, we sum the probabilities of
receiving 0, 1, 2, and 3 customers.
P(X ≤ 3) = P(X = 0) + P(X = 1) + P(X = 2) + P(X = 3)
Calculating each:
• P(X = 0) = (e^(-5) * 5^0) / 0!
= e^(-5) ≈ 0.0067
• P(X = 1) = (e^(-5) * 5^1) / 1!
= (0.0067 * 5) ≈ 0.0335
• P(X = 2) = (e^(-5) * 5^2) / 2!
= (0.0067 * 25) / 2 ≈ 0.0838
• P(X = 3) = (e^(-5) * 5^3) / 3!
= (0.0067 * 125) / 6 ≈ 0.1399
Summing these probabilities:
P(X ≤ 3) ≈ 0.0067 + 0.0335 + 0.0838 + 0.1399 ≈ 0.2639 (or about 26.39%).
#c) If the average number of customers increases to 8 per hour, how would this affect the probability of receiving exactly 4 customers? Briefly explain how this change could impact the coffee shop's staffing decisions. 
If the average number of customers increases to 8, the new λ would be 8.
Calculating the probability of receiving exactly 4 customers with λ = 8:
P(X = 4) = (e^(-8) * 8^4) / 4!
Calculating this gives approximately:
P(X = 4) ≈ 0.0183 (or about 1.83%).
With an increase in the average number of customers, the probability of receiving exactly 4
customers decreases.
