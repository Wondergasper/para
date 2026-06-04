! APG Example: matrix_multiply (Fortran)
! Computes C = A * B for NxN matrices.
! Expected parallelism: polyhedral (outer two loops are independent when C is write-only)
subroutine matrix_multiply(A, B, C, N)
    implicit none
    integer, intent(in)  :: N
    real, intent(in)     :: A(N, N), B(N, N)
    real, intent(out)    :: C(N, N)
    integer :: i, j, k
    real :: tmp
    do i = 1, N
        do j = 1, N
            tmp = 0.0
            do k = 1, N
                tmp = tmp + A(i, k) * B(k, j)
            end do
            C(i, j) = tmp
        end do
    end do
end subroutine matrix_multiply
