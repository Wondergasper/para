! APG Example: dot_product (Fortran)
! Computes the dot product of two arrays A and B.
! Expected parallelism: reduction (sum reduction on s)
real function dot_product_apg(A, B, N)
    implicit none
    real, intent(in) :: A(N), B(N)
    integer, intent(in) :: N
    real :: s
    integer :: i
    s = 0.0
    do i = 1, N
        s = s + A(i) * B(i)
    end do
    dot_product_apg = s
end function dot_product_apg
