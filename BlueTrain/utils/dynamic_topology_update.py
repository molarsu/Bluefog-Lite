

def dynamic_topology_update(dynamic_neighbor_allreduce_gen, args, optimizer):
    if args.dist_optimizer == "neighbor_allreduce":
        send_neighbors, recv_neighbors = next(dynamic_neighbor_allreduce_gen)
        assert len(send_neighbors) == len(recv_neighbors)
        optimizer.dst_weights = {
            r: 1 / (len(send_neighbors) + 1) for r in send_neighbors
        }
        optimizer.src_weights = {
            r: 1 / (len(recv_neighbors) + 1) for r in recv_neighbors
        }
        optimizer.self_weight = 1 / (len(recv_neighbors) + 1)
    else:
        pass