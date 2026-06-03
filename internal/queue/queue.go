package queue

import (
	"context"
	"apg/internal/proto"
)

type Producer interface {
	PublishJob(ctx context.Context, req *proto.JobRequest) error
}

type Consumer interface {
	SubscribeResults(ctx context.Context, handler func(*proto.JobResponse)) error
}
