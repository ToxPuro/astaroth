import matplotlib.pyplot as plt
import csv
def main():
    x = []
    u = []
    a = []
    with open('x.dat', mode ='r')as file:
      csvFile = csv.reader(file)
      for lines in csvFile:
          for elem  in lines:
              x.append(eval(elem))
    with open('u.dat', mode ='r')as file:
      csvfile = csv.reader(file)
      for lines in csvfile:
          for elem  in lines:
              u.append(eval(elem))
    with open('a.dat', mode ='r')as file:
      csvFile = csv.reader(file)
      for lines in csvFile:
          for elem  in lines:
              a.append(eval(elem))

    plt.plot(x,u,label="Simulation")
    plt.scatter(x,a,10,label="Analytical",marker="o", facecolors="none", edgecolors="r")
    plt.xlabel("X-coordinate")
    plt.ylabel("Field value")

    plt.title("Simulation vs. Analytical")
    plt.legend()
    plt.show()
    plt.savefig("Simulation_vs_analytical.png")
if __name__ == "__main__":
    main()

