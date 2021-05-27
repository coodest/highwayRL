# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import src.util.grpc.communication_pb2 as communication__pb2


class CommunicationStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.comm = channel.unary_unary(
                '/memrl.Communication/comm',
                request_serializer=communication__pb2.CommRequest.SerializeToString,
                response_deserializer=communication__pb2.CommReply.FromString,
                )


class CommunicationServicer(object):
    """Missing associated documentation comment in .proto file."""

    def comm(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CommunicationServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'comm': grpc.unary_unary_rpc_method_handler(
                    servicer.comm,
                    request_deserializer=communication__pb2.CommRequest.FromString,
                    response_serializer=communication__pb2.CommReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'memrl.Communication', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Communication(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def comm(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/memrl.Communication/comm',
            communication__pb2.CommRequest.SerializeToString,
            communication__pb2.CommReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
