! APG Example: vector_scale (Fortran)
! A simple affine loop scaling every element of array A by scalar s.
! Expected parallelism: polyhedral (no dependencies between iterations)
subroutine vector_scale(A, s, N)
    implicit none
    real, intent(inout) :: A(N)
    real, intent(in)    :: s
    integer, intent(in) :: N
    integer :: i
    do i = 1, N
        A(i) = A(i) * s
    end do
end subroutine vector_scale
