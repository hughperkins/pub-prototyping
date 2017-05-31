      subroutine hpfn2(a,b,c,d)
      integer a
      real*8 b
      real*8 c(3)
      real*8 d(2,2)
      print *, 'Hello from subroutine testfn2 v0.3', a, b,c(1),c(2),c(3)
      print *,d(1,1),d(1,2)
      print *,d(2,1),d(2,2)
      return
      end

