"""
CITS1401 Project 2
Student Name: Yulin Yu
Student Number: 22743739
Python Version: 3.9.7
"""

# Set default values based on the sample dataset as global values.
adult_id = 0  # Column index for adult ID
land = 1      # Column index for landmark
x = 2         # Column index for X coordinate
y = 3         # Column index for Y coordinate
z = 4         # Column index for Z coordinate


# Calculate the average number
def mean(l):
    return sum(l) / len(l)


# Calculate the Euclidean distance between facial landmarks (OP1)
def cal_ED(data):
    res = {}       # Store unrounded results
    res_round = {} # Store rounded results
    try:
        # Calculate Euclidean distances between predefined facial landmarks
        res["FW"] = ((data['FT_L'][0] - data['FT_R'][0]) ** 2 + (data['FT_L'][1] - data['FT_R'][1]) ** 2 + (data['FT_L'][2] - data['FT_R'][2]) ** 2) ** 0.5
        res["OCW"] = ((data['EX_L'][0] - data['EX_R'][0]) ** 2 + (data['EX_L'][1] - data['EX_R'][1]) ** 2 + (data['EX_L'][2] - data['EX_R'][2]) ** 2) ** 0.5
        res["LEFL"] = ((data['EX_L'][0] - data['EN_L'][0]) ** 2 + (data['EX_L'][1] - data['EN_L'][1]) ** 2 + (data['EX_L'][2] - data['EN_L'][2]) ** 2) ** 0.5
        res["REFL"] = ((data['EN_R'][0] - data['EX_R'][0]) ** 2 + (data['EN_R'][1] - data['EX_R'][1]) ** 2 + (data['EN_R'][2] - data['EX_R'][2]) ** 2) ** 0.5
        res["ICW"] = ((data['EN_L'][0] - data['EN_R'][0]) ** 2 + (data['EN_L'][1] - data['EN_R'][1]) ** 2 + (data['EN_L'][2] - data['EN_R'][2]) ** 2) ** 0.5
        res["NW"] = ((data['AL_L'][0] - data['AL_R'][0]) ** 2 + (data['AL_L'][1] - data['AL_R'][1]) ** 2 + (data['AL_L'][2] - data['AL_R'][2]) ** 2) ** 0.5
        res["ABW"] = ((data['SBAL_L'][0] - data['SBAL_R'][0]) ** 2 + (data['SBAL_L'][1] - data['SBAL_R'][1]) ** 2 + (data['SBAL_L'][2] - data['SBAL_R'][2]) ** 2) ** 0.5
        res["MW"] = ((data['CH_L'][0] - data['CH_R'][0]) ** 2 + (data['CH_L'][1] - data['CH_R'][1]) ** 2 + (data['CH_L'][2] - data['CH_R'][2]) ** 2) ** 0.5
        res["NBL"] = ((data['N'][0] - data['PRN'][0]) ** 2 + (data['N'][1] - data['PRN'][1]) ** 2 + (data['N'][2] - data['PRN'][2]) ** 2) ** 0.5
        res["NH"] = ((data['N'][0] - data['SN'][0]) ** 2 + (data['N'][1] - data['SN'][1]) ** 2 + (data['N'][2] - data['SN'][2]) ** 2) ** 0.5
    except KeyError:
        # If a landmark is missing or corrupted, discard the entire record
        return None, None

    # Round all calculated distances
    for i in res.keys():
        res_round[i] = round(res[i], 4)
    return res, res_round


# Convert data from string to float and create a dictionary for landmarks
def to_dict(data):
    res = {}
    try:
        # Convert X, Y, Z coordinates from string to float
        for i in range(len(data)):
            data[i][x] = float(data[i][x])
            data[i][y] = float(data[i][y])
            data[i][z] = float(data[i][z])
    except ValueError:
        # If data is corrupted or missing, discard the record
        return None
    else:
        # Create a dictionary with landmark names as keys and coordinates as values
        for i in range(len(data)):
            key = data[i][land].upper()  # Landmark name
            value = [data[i][x], data[i][y], data[i][z]]  # X, Y, Z coordinates
            res[key] = value
        return res


# Preprocess the data to extract records for specific adults
def preprocess(data, adults):
    data_processed = []
    for row in data:
        if row[adult_id].upper() == adults.strip().upper():
            data_processed.append(row)
    # Ensure exactly 15 landmarks are available for each adult
    if len(data_processed) == 15:
        return data_processed
    else:
        return None


# Calculate the nominator for cosine similarity (OP2)
def cal_nominator_Simi(dct1, dct2):
    res = 0
    # Sum the product of corresponding facial distances
    for i in dct1.keys():
        res = res + dct1[i] * dct2[i]
    return res


# Calculate the denominator for cosine similarity (OP2)
def cal_denominator_Simi(dct1, dct2):
    res1 = 0
    res2 = 0
    # Calculate the sum of squares for each dictionary
    for i in dct1.keys():
        res1 += dct1[i] ** 2
        res2 += dct2[i] ** 2
    # Return the product of the square roots
    return (res1 ** 0.5) * (res2 ** 0.5)


# Calculate cosine similarity (OP2)
def cal_CS(dct1, dct2):
    try:
        # Calculate cosine similarity and return both raw and rounded values
        return cal_nominator_Simi(dct1, dct2) / cal_denominator_Simi(dct1, dct2), round(
            cal_nominator_Simi(dct1, dct2) / cal_denominator_Simi(dct1, dct2), 4)
    except ZeroDivisionError:
        # Handle cases where the denominator is zero
        print("Denominator's value is zero")
        exit(0)


# Calculate the 5 most similar faces to a reference face (OP3)
def cal_OP3(data, all_faces, face):
    data0 = preprocess(data, face)  # Preprocess data for the reference face
    data0 = to_dict(data0)          # Convert data to dictionary format
    minCS = 2  # Initialize minimum cosine similarity to a high value
    res = []
    for fac in all_faces:
        data1 = preprocess(data, fac)
        if data1 == None:
            continue
        data1 = to_dict(data1)
        if data1 == None:
            continue
        ed1 = cal_ED(data0)[0]
        ed2 = cal_ED(data1)[1]
        if ed2 == None:
            continue
        cosine_simi = cal_CS(ed1, ed2)[0]
        # Find the top 5 similar faces
        if len(res) < 5:
            res.append((fac, cosine_simi))
            if cosine_simi < minCS:
                minCS = cosine_simi
        elif cosine_simi > minCS:
            res.append((fac, cosine_simi))
            remove_list = []
            for i in range(len(res)):
                if res[i][1] == minCS:
                    remove_list.append(res[i])
            remove_list.sort(key=lambda x: x[0])  # Sort by ID
            res.remove(remove_list[-1])           # Remove least similar face
            res.sort(key=lambda x: x[1], reverse=True)  # Sort by similarity
            minCS = res[-1][1]  # Update minimum cosine similarity
    for i in range(len(res)):
        res[i] = (res[i][0], round(res[i][1], 4))  # Round results
    return res


# Calculate the average of distances for the closest 5 faces (OP4)
def cal_OP4(data, faces):
    res = {}
    for face in faces:
        data0 = preprocess(data, face)
        data0 = to_dict(data0)
        facial_distance = cal_ED(data0)[0]
        for key in facial_distance.keys():
            if key in res.keys():
                res[key].append(facial_distance[key])
            else:
                res[key] = [facial_distance[key]]
    # Calculate the mean of each facial distance and round the result
    for i in res.keys():
        res[i] = round(mean(res[i]), 4)
    return res


# Main function to process the CSV file and compute OP1, OP2, OP3, and OP4
def main(csvfile, adults):
    global adult_id, land, x, y, z
    filename = csvfile.strip()
    adults[0] = adults[0].strip().upper()
    adults[1] = adults[1].strip().upper()
    if adults[0] == adults[1]:
        print("The two input faces are the same.")
        exit(0)
    try:
        # Open the data file
        facial_data = open(filename, 'r')
    except FileNotFoundError:
        print("The file can't be found")
        return (None, None, None, None)

    # Read and parse the CSV file
    data = facial_data.read()
    facial_data.close()

    data = data.strip().split('\n')  # Group data by rows and convert to a list
    for i in range(len(data)):
        data[i] = data[i].split(',')
    # Dynamically identify column positions based on headers
    for item in range(len(data[0])):
        if data[0][item].strip().upper() == 'ADULTID':
            adult_id = item
        if data[0][item].strip().upper() == 'LANDMARK':
            land = item
        if data[0][item].strip().upper() == 'X':
            x = item
        if data[0][item].strip().upper() == 'Y':
            y = item
        if data[0][item].strip().upper() == 'Z':
            z = item

    # Get the list of all face IDs
    all_faces = list(set([data[i][adult_id].upper() for i in range(1, len(data))]))

    # Remove the two input faces from the list
    all_faces.remove(adults[0])
    all_faces.remove(adults[1])

    # Preprocess data for the two input faces
    data0 = preprocess(data, adults[0])
    if data0 == None:
        print("Face 1 ID loss")
        return None, None, None, None
    data0 = to_dict(data0)
    data1 = preprocess(data, adults[1])
    if data1 == None:
        print("Face 2 ID loss")
        return None, None, None, None
    data1 = to_dict(data1)
    if data0 == None or data1 == None:
        print("The XYZ data for the two input faces is missing")
        return None, None, None, None
    if ("" in data0.keys()) or ("" in data1.keys()):
        print("The Landmark data for the two input faces is missing")
        return None, None, None, None

    # Calculate OP1 (Euclidean distances)
    OP1 = [cal_ED(data0)[1], cal_ED(data1)[1]]

    # Calculate OP2 (Cosine similarity)
    OP2 = cal_CS(cal_ED(data0)[0], cal_ED(data1)[0])[1]

    # Calculate OP3 (Top 5 similar faces)
    OP3 = [cal_OP3(data, all_faces, adults[0]), cal_OP3(data, all_faces, adults[1])]

    # Extract the top 5 faces for both input faces
    faces1 = [OP3[0][i][0] for i in range(len(OP3[0]))]
    faces2 = [OP3[1][i][0] for i in range(len(OP3[1]))]

    # Calculate OP4 (Average distances of closest faces)
    OP4 = [cal_OP4(data, faces1), cal_OP4(data, faces2)]
    
    return (OP1, OP2, OP3, OP4)


# Example of how the main function can be called
if __name__ == '__main__':
    OP1, OP2, OP3, OP4 = main('sample_face_data1.csv', ['   P1283  ', ' r7033   '])
    print(OP1)
    print(OP2)
    print(OP3)
    print(OP4)
