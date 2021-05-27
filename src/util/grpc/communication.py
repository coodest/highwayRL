import os
from src.module.agent.policy import Policy
import grpc

import src.util.grpc as g
from concurrent import futures


class Client:
    def __init__(self, server_address):
        channel = grpc.insecure_channel(server_address)
        self.stub = g.communication_pb2_grpc.CommunicationStub(channel)

    def comm(self, last_obs, action, obs, reward, add):
        reply = self.stub.comm(g.communication_pb2.CommRequest(
            last_obs=last_obs,
            action=action,
            obs=obs,
            reward=reward,
            add=add
        ))

        return reply.action


class Server:
    def __init__(self, server_address):
        self.server_address = server_address

    def start(self):
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
        g.communication_pb2_grpc.add_CommunicationServicer_to_server(Policy(), server)
        server.add_insecure_port(self.server_address)
        server.start()
        server.wait_for_termination()
