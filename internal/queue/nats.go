package queue

import (
	"context"
	"fmt"
	"log"

	"apg/internal/proto"
	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	googleproto "google.golang.org/protobuf/proto"
)

type NATSQueue struct {
	nc *nats.Conn
	js jetstream.JetStream
}

var APGStreamSubjects = []string{"apg.jobs.*", "apg.results"}

func NewNATSQueue(url string) (*NATSQueue, error) {
	nc, err := nats.Connect(url)
	if err != nil {
		return nil, fmt.Errorf("nats connect: %w", err)
	}

	js, err := jetstream.New(nc)
	if err != nil {
		return nil, fmt.Errorf("jetstream init: %w", err)
	}

	// Create stream
	ctx := context.Background()
	_, err = js.CreateOrUpdateStream(ctx, jetstream.StreamConfig{
		Name:     "APG",
		Subjects: APGStreamSubjects,
	})
	if err != nil {
		return nil, fmt.Errorf("create stream: %w", err)
	}

	return &NATSQueue{nc: nc, js: js}, nil
}

func (q *NATSQueue) PublishJob(ctx context.Context, req *proto.JobRequest, isHeavy bool) error {
	data, err := googleproto.Marshal(req)
	if err != nil {
		return err
	}

	subj := "apg.jobs.quick"
	if isHeavy {
		subj = "apg.jobs.heavy"
	}

	_, err = q.js.Publish(ctx, subj, data)
	return err
}

func (q *NATSQueue) SubscribeResults(ctx context.Context, handler func(*proto.JobResponse)) error {
	c, err := q.js.CreateConsumer(ctx, "APG", jetstream.ConsumerConfig{
		Durable:       "orchestrator",
		FilterSubject: "apg.results",
	})
	if err != nil {
		return err
	}

	iter, err := c.Messages()
	if err != nil {
		return err
	}

	go func() {
		for {
			msg, err := iter.Next()
			if err != nil {
				log.Printf("nats consume error: %v", err)
				return
			}

			var resp proto.JobResponse
			if err := googleproto.Unmarshal(msg.Data(), &resp); err != nil {
				log.Printf("nats unmarshal error: %v", err)
				msg.Term()
				continue
			}

			handler(&resp)
			msg.Ack()
		}
	}()

	return nil
}

func (q *NATSQueue) Close() {
	q.nc.Close()
}
