import os
import csv
import glob

def summarize_results():
    # Set the directory to ft_evaluation_results
    directory = 'ft_evaluation_results'
    
    # Change the current working directory
    os.chdir(directory)
    
    # Get all CSV files in the ft_evaluation_results directory
    csv_files = glob.glob('*.csv')
    
    # Prepare the output file
    output_file = 'summary_results.txt'
    
    with open(output_file, 'w') as outfile:
        for csv_file in csv_files:
            outfile.write(f"Results from {csv_file}:\n")
            
            with open(csv_file, 'r') as infile:
                csv_reader = csv.reader(infile)
                for row in csv_reader:
                    outfile.write(','.join(row) + '\n')
            
            outfile.write('\n')  # Add a blank line between files
    
    print(f"Summary has been written to {os.path.join(directory, output_file)}")
    print("Original CSV files have been preserved.")

if __name__ == "__main__":
    summarize_results()


