    
#联邦聚类算法    
def fed_fuc(recv_fin,mpi_nums):  
    now_parameters = recv_fin[0]
    recv_fin = recv_fin[1:]
    for now_dict  in recv_fin :
        for key in now_dict:
            now_parameters[key] += now_dict[key]
    for key in now_parameters:
        now_parameters[key] = now_parameters[key]/mpi_nums
    return now_parameters