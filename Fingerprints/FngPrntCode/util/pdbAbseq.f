       program pdbseq
       character*80 line
       character*6 label1
       character*3 lkeepH
       logical lkH,lstlab
       character*3 seqstrng(2000), THREEreslab(20) 
       character*1 one_letterC(2000), ONEreslab(20)
       character*1 CARACTER(26), monmer(26), CARES 
       character*1 monIn,monOld,monstrg(2000)
       integer lastRmon(0:26),ifrstRmon(0:26) ! maximum 26 monomers
        character*5 LABRES, LABRESOLD


       
       
       data THREEreslab /'ALA','ASP','ASN','ARG','GLU','GLN','HIS' 
     & ,'LEU' ,'LYS','CYS','TRP','VAL','SER','ILE','MET','TYR' 
     & ,'THR','PRO','GLY','PHE'/
       data ONEreslab /'A','D','N','R','E','Q','H','L','K','C','W','V'
     &  ,'S','I','M','Y','T','P','G','F'/
       data CARACTER /'A','B','C','D','E','F','G','H','I','J','K',
     &   'L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'/
!ATOM      2  N   LYS A   2       0.000   0.000   0.000 -4.28 -0.33
!                     ^
         
!      write(*,'(a40)')'ENTER SHIFT ON RESIDUE NUMBER'
       imon=0
       monOld =" "
       monIn=" "
       ifrstRmon(1)=1
       inumer=0
       lstres=-9999999
       lstlab=.false.
       lkH=.true.
  100  continue
       read(10,'(a)', END= 500) line
!      write(*,'(A80)') line(1:80)
       read(line(1:6),'(a6)') label1
!      write(*,'(A6)') label1
       if (label1(1:4).eq.'ATOM') then
          read(line(22:22),'(A1)') monIn
          if (monIn .eq. " ") then
              imon=1
              monmer(imon)="X"
          elseif (monIn .ne. monOld) then
              imon=imon+1
              monmer(imon)=monIn
              if(monOld.eq." ") then
                ifrstRmon(imon)=1
              else
                lastRmon(imon-1)=inumer
                ifrstRmon(imon)=inumer+1
              endif
              monOld=monIn
          endif
          read(line(23:26),'(I4)') IRES

          read(line(27:27),'(A1)') CARES
          if(CARES .ne. ' ') then
             read(line(23:27),'(A5)') LABRES
             if(LABRES .ne. LABRESOLD) then
                do i=1,26
                   if(CARES .eq. CARACTER(i) ) then
                      lstlab=.true.
                      inumer= inumer+1
                      seqstrng(inumer)=line(18:20)
                      monstrg(inumer)=monIn
                      LABRESOLD=LABRES
                      goto 444
                   endif
                enddo
                write(*,*)'ERROR in residue-number label',line(23:27)
                stop 443
444          continue
             endif
          else
             if (lstlab) then
                lstlab=.false.
                LABRESOLD=""
                inumer= inumer+1
                seqstrng(inumer)=line(18:20)
                monstrg(inumer)=monIn
             else
                if (ires. ne. lstres) then
                  inumer= inumer+1
                  seqstrng(inumer)=line(18:20)
                  monstrg(inumer)=monIn
                endif
             endif
          endif
          lstres=ires

          inumer0= inumer
          if( (line(14:14).ne.'H') .or. lkH)  THEN
! exclude H atoms
!            write(9,'(A22,I4,A1,A53)') 
!    &            line(1:22),inumer0,' ',line(28:80)
             write(9,'(A22,I4,A1,A39)') 
     &            line(1:22),inumer0,' ',line(28:66)
          endif
       elseif ((label1(1:6).eq.'SSBOND') .or.
     &                       (label1(1:3).eq.'TER')) then
          write(9,'(A66)') line(1:66)
! Dont write anything for HETATM or CONECT
       elseif ((label1(1:6).eq.'HETATM').or.
     &                       (label1(1:6).eq.'CONECT')) then
! else, dont write anything
       endif
       goto 100
  500  continue
       
! When the PDB file ends
!save number of last residue in the monomer  
       lastRmon(imon)=inumer0

       do m=1, imon
          ifst= ifrstRmon(m)
          ilast= lastRmon(m)
          if (m.lt.27)write(12,'(a1,a1)') ">",monmer(m)
          if (m.ge.27) then
              write(12,'(a16,i10)') "too many chains ",m
            stop 3333
          endif
          do i=ifst,ilast
             do j=1,20
                if(seqstrng(i) .eq. THREEreslab(j) ) then
                  one_letterC(i)= ONEreslab(j)
                  goto 555
                endif
             enddo
             write(*,*)'ERROR assigning one-letter code to residue ',i
             stop 553
555          continue
          enddo
          write(12,'(2000(a1))') (one_letterC(i),i=ifst,ilast)
       enddo
       stop
       end
