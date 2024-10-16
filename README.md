## Project Overview
This project involves analysing 3D Euclidean distances between significant facial landmarks to perform face recognition. Using a dataset containing 3D coordinates of 15 facial landmarks for various individuals, the program calculates facial distances, compares the similarity between two faces, and identifies the five most similar faces from the dataset.

## Features
- **Euclidean Distance Calculation**: Computes 3D Euclidean distances between 10 specified landmark pairs for given faces.
- **Cosine Similarity Calculation**: Calculates cosine similarity between the facial distances of two individuals.
- **Face Similarity Ranking**: Identifies the five most similar faces to a reference face based on cosine similarity.
- **Flexible Input Handling**: Dynamically processes CSV files with varying column orders.
- **Data Validation**: Discards corrupt or missing landmark records during processing.

## Inputs
1. **CSV File**: The input CSV file contains de-identified 3D landmark coordinates (X, Y, Z) for each individual.
   - Columns include `AdultID`, `Landmark`, and the 3D coordinates `X`, `Y`, `Z`.
2. **Adult IDs**: A list of two adult IDs (case-insensitive) whose facial data will be compared.

## Outputs
The `main` function returns the following four outputs:
1. **OP1**: A list of two dictionaries containing the 3D Euclidean distances for the two faces (for the 10 specified distances).
2. **OP2**: The cosine similarity between the two faces based on their facial distances.
3. **OP3**: A list of two tuples showing the cosine similarity between each face and the five most similar faces (excluding each other).
4. **OP4**: A list of two dictionaries showing the average facial distances for the closest five faces to each reference face.

## Example Usage
```python
# examples of how you can call the program
OP1, OP2, OP3, OP4 = main(sample_face_data.csv, ['R7033', 'P1283'])

# Print the results
print("OP1:", OP1)
print("OP2:", OP2)
print("OP3:", OP3)
print("OP4:", OP4)
```

## How to Run
- Clone this repository
- Run the script by providing the necessary inputs (csvfile, and ).
