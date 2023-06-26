import csv
import matplotlib.pyplot as plt

# Read data from CSV file
data = []
with open('C:/Data/global/Earth-Sun_distance.csv', 'r') as file:
    reader = csv.reader(file, delimiter=';')
    for row in reader:
        data.append(row)

# Extract day of the year and distance values
days = []
distances = []
for row in data[2:-10]:  # Skip header and footer rows
    for i in range(0, len(row), 2):
        day = row[i].strip()
        distance = row[i + 1].replace(',', '.').strip()
        if day and distance:  # Skip empty entries
            days.append(int(day))
            distances.append(float(distance))

# Sort the days and distances in ascending order
days, distances = zip(*sorted(zip(days, distances)))


# Plot the data
plt.plot(days, distances)
plt.xlabel('Day of the Year')
plt.ylabel('Earth-Sun Distance (AU)')
plt.title('Earth-Sun Distance Variation')
plt.grid(True)
plt.show()