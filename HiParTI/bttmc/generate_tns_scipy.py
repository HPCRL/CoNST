import numpy as np
import argparse
import os
import subprocess

def generate_ttmc_dense_matrices_m(B, K, dir_path):   
    #m = np.random.rand(B,K)
    file_path = dir_path + "/m.tns"
    with open(file_path, 'w') as gfile:
        gfile.write("{} {} {}\n".format(2, B, K))
    with open(file_path, "a") as gfile:
        for x in range(B):
            for y in range(K):
                gfile.write("{} {} {}\n".format(x+1, y+1, 1.0))
    print("generate_ttmc_dense_matrices_m done", flush=True)
    return file_path
                
def generate_ttmc_dense_matrices_n(A, J, dir_path):   
    #n = np.random.rand(A, J)
    file_path = dir_path + "/n.tns"
    with open(file_path, 'w') as gfile:
        gfile.write("{} {} {}\n".format(2, A, J))
    with open(file_path, "a") as gfile:
        for x in range(A):
            for y in range(J):
                gfile.write("{} {} {}\n".format(x+1, y+1, 1.0))
    print("generate_ttmc_dense_matrices_n done", flush=True)
    return file_path
                
def generate_ttmc_dense_matrices_x(A, I, dir_path):  
    #x = np.random.rand(A, I)
    file_path = dir_path + "/x.tns"
    with open(file_path, 'w') as gfile:
        gfile.write("{} {} {}\n".format(2, A, I))
    with open(file_path, "a") as gfile:
        for x in range(A):
            for y in range(I):
                gfile.write("{} {} {}\n".format(x+1, y+1, 1.0))
    print("generate_ttmc_dense_matrices_x done", flush=True)
    return file_path
 
def generate_ttmc_dense_matrices_y(B, J, dir_path):     
    #y = np.random.rand(B, J)
    file_path = dir_path + "/y.tns"
    with open(file_path, 'w') as gfile:
        gfile.write("{} {} {}\n".format(2, B, J))
    with open(file_path, "a") as gfile:
        for x in range(B):
            for y in range(J):
                gfile.write("{} {} {}\n".format(x+1, y+1, 1.0))
    print("generate_ttmc_dense_matrices_y done", flush=True)
    return file_path

"""
mode 1
((I*M)*N)

mode 2
((I*M)*X)

mode 3
((I*Y)*X)
"""

def get_meta_from_tns(tns_file):
    with open(tns_file, "r") as gfile:
        first_line = gfile.readline()
        if first_line[-1] == "\n":
            first_line = first_line[:-1]
        meta = first_line.split(" ")
        meta = [int(x) for x in meta if x != ""]
        print("meta info  : ", meta, flush=True)
        assert meta[0]  == 3
        assert len(meta) == meta[0] + 1
        
    return meta[1], meta[2], meta[3]  

def call_sparta_m1(i_path, m_path, n_path, config_path, bash_script_path):  
    print(f"sparta m1 -i {i_path} -m {m_path} -n {n_path}", flush=True)
    script_arguments = [i_path, m_path, n_path, config_path]
    print(script_arguments)
    result = subprocess.run(['bash', bash_script_path] + script_arguments, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    # Print the error, if any
    if result.stderr:
        print("Error:")
        print(result.stderr)
   

def call_sparta_m2(i_path, m_path, x_path, config_path, bash_script_path):  
    print(f"sparta m2 -i {i_path} -m {m_path} -x {x_path}", flush=True)  
    script_arguments = [i_path, m_path, x_path, config_path]
    result = subprocess.run(['bash', bash_script_path] + script_arguments, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(result.stdout)
    # Print the error, if any
    if result.stderr:
        print("Error:")
        print(result.stderr)
    

def call_sparta_m3(i_path, x_path, y_path, config_path, bash_script_path):
    print(f"sparta m3 -i {i_path} -x {x_path} -y {y_path}", flush=True)  
    script_arguments = [i_path, y_path, x_path, config_path]
    result = subprocess.run(['bash', bash_script_path] + script_arguments, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Print the error, if any
    print(result.stdout)
    if result.stderr:
        print("Error:")
        print(result.stderr)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate TNS files for TTMC')
    # Add optional argument
    #parser.add_argument('--mode', '-m', help='mode of the tensor', type=int, default=-1)
    parser.add_argument('--tns', '-t', help='tns file')
    parser.add_argument('--conf', '-c', help='config file')
    parser.add_argument('--bash', '-b', help='bash script file')
    # Parse the arguments
    args = parser.parse_args()
    
    #compute_mode = int(args.mode)
    tns_file = args.tns
    config_file = args.conf
    bash_script_path = args.bash
    print("\n--------------------------------------------" )
    #print(f'compute_mode: {compute_mode}')
    print(f'tns_file: {tns_file}')
    print(f'config_file_dir: {config_file}')
    print(f'bash_script_path: {bash_script_path}')
    print("--------------------------------------------" )
    
    if "nell-1" in tns_file:
        A = B = 16
    elif "flickr-3d" in tns_file:
        A = B = 16
    elif "nell-2" in tns_file:
        A = B = 50
    elif "vast-2015-mc1-3d" in tns_file:
        A = B = 50
    else:
        raise ValueError("Unknown tns file")
    
    I, J, K = get_meta_from_tns(tns_file)
    directory_path = os.path.dirname(tns_file)
    
    m_path, n_path, x_path, y_path = None, None, None, None
    config_file_map={1: config_file+"conf-m1.txt",
                     2: config_file+"conf-m2.txt",
                     3: config_file+"conf-m3.txt"}
    
    for compute_mode in range(1, 4):
        config_file = config_file_map[compute_mode]
        if compute_mode == 1:
            print(f'================ Start {tns_file } mode 1 ================')
            m_path = generate_ttmc_dense_matrices_m(B, K, directory_path)
            n_path = generate_ttmc_dense_matrices_n(A, J, directory_path)
            call_sparta_m1(tns_file, m_path, n_path, config_file, bash_script_path)
        elif compute_mode == 2:
            print(f'================ Start {tns_file } mode 2 ================')
            print(f'n_path: {n_path}')
            if os.path.exists(n_path):
                os.remove(n_path)
                print(f'The file {n_path} has been deleted.')
            else:
                print(f'The file {n_path} does not exist.')
            x_path = generate_ttmc_dense_matrices_x(A, I, directory_path)
            call_sparta_m2(tns_file, m_path, x_path, config_file, bash_script_path)
        elif compute_mode == 3:
            print(f'================ Start {tns_file } mode 3 ================')
            if os.path.exists(m_path):
                os.remove(m_path)
                print(f'The file {m_path} has been deleted.')
            else:
                print(f'The file {m_path} does not exist.')
            y_path = generate_ttmc_dense_matrices_y(B, J, directory_path)   
            call_sparta_m3(tns_file, x_path, y_path, config_file, bash_script_path)
            print("Cleaning up...")
            os.remove(x_path) 
            print(f'The file {x_path} has been deleted.')
            os.remove(y_path) 
            print(f'The file {y_path} has been deleted.')
    print(f'================ Done {tns_file } ================')
    
    #manual gen tensor
    # if compute_mode == 1:
    #         print(f'================ Start {tns_file } mode 1 ================')
    #         m_path = generate_ttmc_dense_matrices_m(B, K, directory_path)
    #         n_path = generate_ttmc_dense_matrices_n(A, J, directory_path)
    #         # call_sparta_m1(tns_file, m_path, n_path, config_file, bash_script_path)
    # elif compute_mode == 2:
    #     print(f'================ Start {tns_file } mode 2 ================')
    #     m_path = generate_ttmc_dense_matrices_m(B, K, directory_path)
    #     x_path = generate_ttmc_dense_matrices_x(A, I, directory_path)
    #     #call_sparta_m2(tns_file, m_path, x_path, config_file, bash_script_path)
    #     os.remove(m_path) 
    #     print(f'The file {m_path} has been deleted.')
    #     os.remove(x_path) 
    #     print(f'The file {x_path} has been deleted.')
    # elif compute_mode == 3:
    #     x_path = generate_ttmc_dense_matrices_x(A, I, directory_path)
    #     y_path = generate_ttmc_dense_matrices_y(B, J, directory_path)   
    #     #call_sparta_m3(tns_file, x_path, y_path, config_file, bash_script_path)
       
    #     os.remove(x_path) 
    #     print(f'The file {x_path} has been deleted.')
    #     os.remove(y_path) 
    #     print(f'The file {y_path} has been deleted.')
    # print(f'================ Done {tns_file } ================')
    
