def is_binary(occurences):
    if set(occurences) == {"0","1"} or set(occurences) =={0,1}:
        return True
    else:
        return False

# Encoding with RLE :
def RLE(occurences):
    """
    Why:
    Initialize an empty list to store the result of the Run-Length Encoding (RLE) transformation.
    line code:
    RLE = []
    """
    RLE = []
    
    """
    Why:
    Check if the input occurrences are binary (specific case handling).
    line code:
    if is_binary(occurences):
    """
    if is_binary(occurences):
        """
        Why:
        Handle cases where binary occurrences start with the value 1. Append a leading 0 to the RLE result and process the subsequent counts.
        line code:
        if occurences[0][0] == 1:
        """
        if occurences[0][0] == 1:
            """
            Why:
            Append 0 to the RLE list as a prefix for binary sequences starting with 1.
            line code:
            RLE.append(0)
            """
            RLE.append(0)
            """
            Why:
            Iterate through the occurrences and append the integer counts to the RLE list.
            line code:
            for i in occurences:
                RLE.append(int(i[1]))
            """
            for i in occurences:
                RLE.append(int(i[1]))
        else:
            """
            Why:
            For binary sequences not starting with 1, directly append the integer counts.
            line code:
            for i in occurences:
                RLE.append(int(i[1]))
            """
            for i in occurences:
                RLE.append(int(i[1]))
    else:
        """
        Why:
        For non-binary occurrences, append each pair of values (element and its count) to the RLE list.
        line code:
        for i in occurences:
            RLE.append(i[0])
            RLE.append(i[1])
        """
        for i in occurences:
            RLE.append(i[0])
            RLE.append(i[1])
    
    """
    Why:
    Return the completed RLE list after processing all occurrences.
    line code:
    return RLE
    """
    return RLE

            
            
# Decoding with RLE :
def DecodeRLE(rle):
    # Create an empty list called 'original' to store the decoded sequence.
    original = []
    # Create an empty string 'text' to accumulate decoded characters (if the RLE encodes text).
    text = ""
    
    # Check if the first element of the RLE sequence is 0.
    # This branch appears to be used when the RLE encodes a binary sequence (0’s and 1’s)
    # in a specific alternating pattern.
    if rle[0] == 0:
        # Initialize a flag variable to 1. This flag will toggle between 1 and 0.
        flag = 1
        # Loop through each element 'i' in the RLE sequence.
        for i in rle:
            # If flag is 1, then add 'i' number of 1's to the original list.
            if flag == 1:
                # For each count (i times), append the value 1.
                for j in range(i):
                    original.append(1)
                # Switch the flag to 0 so the next count will append 0's.
                flag = 0
            else:
                # If flag is 0, then add 'i' number of 0's to the original list.
                for j in range(i):
                    original.append(0)
                # Switch the flag back to 1 for the next iteration.
                flag = 1
        # After processing all counts, return the decoded binary sequence.
        return original

    # If the first element is of type 'str', then the RLE is assumed to encode text.
    elif type(rle[0]) == str:
        # Process the RLE list in pairs: each pair consists of a character and its repetition count.
        # The loop goes from 0 to len(rle)-1, stepping by 2 each time.
        for i in range(0, len(rle) - 1, 2):
            # Multiply the character (rle[i]) by the number (rle[i+1]) to create a repeated substring,
            # and concatenate it to 'text'.
            text += rle[i] * rle[i+1]
        # Return the reconstructed text string.
        return text