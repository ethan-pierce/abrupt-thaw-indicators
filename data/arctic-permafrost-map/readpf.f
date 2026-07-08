      program readpf
C     sample program to read permafrost arrays

      implicit none

      integer width, height

C     Nh grid dimensions
      parameter (width = 1441, height = 1441)

C     Nl grid dimensions
CCC      parameter (width = 721, height = 721)

C     1/2 x 1/2 deg grid dimensions
CCC      parameter (width = 720, height = 360)

      character*1 pf(width, height)

      integer row, col, x, y

C     open file and read data
      open(unit=1, file='NhIPA.byte', form='system')

      read(1) pf

C     query grid
 100  continue

C     x,y start at 0,0 in lower left corner
C     rdpix is an IDL routine used for testing

      print *, 'enter x y from rdpix'
      read *, x, y
      col = height-y
      row = x+1

      print '(a,i4,a,i4,a,i4)', 
     1     'col=', col, ', row=', row, ', value=', pf(row, col)

      goto 100

      end

