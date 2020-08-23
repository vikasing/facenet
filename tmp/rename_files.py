import os


# Function to rename multiple files
def main():
    dir = "/home/vik/data/Photos/vikash/"
    for count, filename in enumerate(os.listdir(dir)):
        dst = "Vikash_Singh_00" + str(count) + ".jpg"
        src = dir + filename
        dst = dir + dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)

    # Driver Code


if __name__ == '__main__':
    # Calling main() function
    main()