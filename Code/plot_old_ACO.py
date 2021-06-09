import ACO_Improved
import ACO_Traditional
import matplotlib.pyplot as plt

traditional = ACO_Traditional.main(False)
improved = ACO_Improved.main(False)
print(traditional)
print(improved)
plt.xlabel('Iterations')
plt.ylabel('Best Fitness')
plt.plot(traditional, label = "Traditional ACO")
plt.plot(improved, label = "Improved ACO")
plt.title("Convergence on 20x20 Grid")
plt.legend()
plt.show()