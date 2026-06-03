package queue

import "testing"

func TestAPGStreamSubjectsCoverWorkerJobSubjects(t *testing.T) {
	want := []string{"apg.jobs.*", "apg.results"}
	if len(APGStreamSubjects) != len(want) {
		t.Fatalf("APGStreamSubjects length = %d, want %d", len(APGStreamSubjects), len(want))
	}
	for i, subject := range want {
		if APGStreamSubjects[i] != subject {
			t.Fatalf("APGStreamSubjects[%d] = %q, want %q", i, APGStreamSubjects[i], subject)
		}
	}
}
