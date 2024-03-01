# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from airbot.lm.rpc import service_pb2 as airbot_dot_lm_dot_rpc_dot_service__pb2


class InstructionServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.extract = channel.unary_unary(
            '/LanguageModel.InstructionService/extract',
            request_serializer=airbot_dot_lm_dot_rpc_dot_service__pb2.ExtractRequest.SerializeToString,
            response_deserializer=airbot_dot_lm_dot_rpc_dot_service__pb2.ExtractResponse.FromString,
        )


class InstructionServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def extract(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_InstructionServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'extract':
            grpc.unary_unary_rpc_method_handler(
                servicer.extract,
                request_deserializer=airbot_dot_lm_dot_rpc_dot_service__pb2.ExtractRequest.FromString,
                response_serializer=airbot_dot_lm_dot_rpc_dot_service__pb2.ExtractResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler('LanguageModel.InstructionService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class InstructionService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def extract(request,
                target,
                options=(),
                channel_credentials=None,
                call_credentials=None,
                insecure=False,
                compression=None,
                wait_for_ready=None,
                timeout=None,
                metadata=None):
        return grpc.experimental.unary_unary(request, target, '/LanguageModel.InstructionService/extract',
                                             airbot_dot_lm_dot_rpc_dot_service__pb2.ExtractRequest.SerializeToString,
                                             airbot_dot_lm_dot_rpc_dot_service__pb2.ExtractResponse.FromString, options,
                                             channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)


class ImageServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.detect = channel.unary_unary(
            '/LanguageModel.ImageService/detect',
            request_serializer=airbot_dot_lm_dot_rpc_dot_service__pb2.DetectRequest.SerializeToString,
            response_deserializer=airbot_dot_lm_dot_rpc_dot_service__pb2.DetectResponse.FromString,
        )
        self.segment = channel.unary_unary(
            '/LanguageModel.ImageService/segment',
            request_serializer=airbot_dot_lm_dot_rpc_dot_service__pb2.SegmentRequest.SerializeToString,
            response_deserializer=airbot_dot_lm_dot_rpc_dot_service__pb2.SegmentResponse.FromString,
        )


class ImageServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def detect(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def segment(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ImageServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        'detect':
            grpc.unary_unary_rpc_method_handler(
                servicer.detect,
                request_deserializer=airbot_dot_lm_dot_rpc_dot_service__pb2.DetectRequest.FromString,
                response_serializer=airbot_dot_lm_dot_rpc_dot_service__pb2.DetectResponse.SerializeToString,
            ),
        'segment':
            grpc.unary_unary_rpc_method_handler(
                servicer.segment,
                request_deserializer=airbot_dot_lm_dot_rpc_dot_service__pb2.SegmentRequest.FromString,
                response_serializer=airbot_dot_lm_dot_rpc_dot_service__pb2.SegmentResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler('LanguageModel.ImageService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class ImageService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def detect(request,
               target,
               options=(),
               channel_credentials=None,
               call_credentials=None,
               insecure=False,
               compression=None,
               wait_for_ready=None,
               timeout=None,
               metadata=None):
        return grpc.experimental.unary_unary(request, target, '/LanguageModel.ImageService/detect',
                                             airbot_dot_lm_dot_rpc_dot_service__pb2.DetectRequest.SerializeToString,
                                             airbot_dot_lm_dot_rpc_dot_service__pb2.DetectResponse.FromString, options,
                                             channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)

    @staticmethod
    def segment(request,
                target,
                options=(),
                channel_credentials=None,
                call_credentials=None,
                insecure=False,
                compression=None,
                wait_for_ready=None,
                timeout=None,
                metadata=None):
        return grpc.experimental.unary_unary(request, target, '/LanguageModel.ImageService/segment',
                                             airbot_dot_lm_dot_rpc_dot_service__pb2.SegmentRequest.SerializeToString,
                                             airbot_dot_lm_dot_rpc_dot_service__pb2.SegmentResponse.FromString, options,
                                             channel_credentials, insecure, call_credentials, compression,
                                             wait_for_ready, timeout, metadata)
