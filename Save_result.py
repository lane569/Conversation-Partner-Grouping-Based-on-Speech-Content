#!/usr/bin/python
# -*- coding: UTF-8 -*-
import traceback

def save_similarity_matrix(matrix, output_path):
    try:

        outfile = open( output_path, "w" )

        for row_list in matrix:
            line = ""
            for value in row_list:
                line += ( str(value) + ',' )
            outfile.write(line + '\n')

        outfile.close()

        #print("[INFO]:save_similarity_matrix is finished!")

    except Exception as e:
        print(traceback.print_exc())
