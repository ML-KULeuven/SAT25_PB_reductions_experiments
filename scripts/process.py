import cactus
import csv_inconsistencies

if __name__ == '__main__':
    print("Checking inconsistencies...")
    csv_inconsistencies.main()
    print("Plotting results...")
    cactus.main()