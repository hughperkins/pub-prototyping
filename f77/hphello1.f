

      program main
c      implicit none
c      write ( *, '(a)' ) ' Hello, world!'
      integer c(3)
      real*8 d
      real*8 e(3)
      real*8 f(2,2)
      print *, 'hello world', 5
      c(1) = 12
      C(2) = 14
      c(3) = 18
      call testfn(3, c)
      d = 2.7
      e(1) = 1.23
      e(2) = 7.56
      e(3) = 8.35
      f(1,1)= 3
      f(1,2)=5
      f(2,1) = 4
      f(2,2) = 7
      call hpfn2(7, d, e, f )
c      stop
      end

      subroutine testfn(a, b)
      integer a
      integer b(3)
c      write ( *, '(a)' ) 'Hello from subroutine test',32
      print *, 'hello from subroutine test', a, b(1),b(2),b(3)
      return
      end

