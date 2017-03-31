Search.setIndex({docnames:["api/advanced_tables","api/api","api/base","api/basic_tables","api/configuration","api/filetypes/cube","api/filetypes/xyz","api/js/ao","api/js/app","api/js/field","api/js/gaussian","api/js/gtf","api/js/harmonics","api/js/universe","api/orbital_tables","dev/overview","index","install","overview"],envversion:51,filenames:["api/advanced_tables.rst","api/api.rst","api/base.rst","api/basic_tables.rst","api/configuration.rst","api/filetypes/cube.rst","api/filetypes/xyz.rst","api/js/ao.rst","api/js/app.rst","api/js/field.rst","api/js/gaussian.rst","api/js/gtf.rst","api/js/harmonics.rst","api/js/universe.rst","api/orbital_tables.rst","dev/overview.rst","index.rst","install.rst","overview.rst"],objects:{"exatomic.atom":{Atom:[3,1,1,""],Frequency:[3,1,1,""],ProjectedAtom:[3,1,1,""],UnitAtom:[3,1,1,""],VisualAtom:[3,1,1,""]},"exatomic.atom.Atom":{from_small_molecule_data:[3,2,1,""],get_atom_labels:[3,3,1,""],get_element_masses:[3,3,1,""],last_frame:[3,4,1,""],nframes:[3,4,1,""],to_xyz:[3,3,1,""],unique_atoms:[3,4,1,""]},"exatomic.atom.VisualAtom":{from_universe:[3,2,1,""]},"exatomic.basis":{BasisSetOrder:[14,1,1,""],GaussianBasisSet:[14,1,1,""],Overlap:[14,1,1,""]},"exatomic.basis.Overlap":{from_column:[14,2,1,""]},"exatomic.container":{Meta:[2,1,1,""],Universe:[2,1,1,""],basis_function_contributions:[2,5,1,""],concat:[2,5,1,""]},"exatomic.container.Meta":{atom:[2,4,1,""],atom_two:[2,4,1,""],basis_set_order:[2,4,1,""],contribution:[2,4,1,""],density:[2,4,1,""],excitation:[2,4,1,""],field:[2,4,1,""],frame:[2,4,1,""],frequency:[2,4,1,""],molecule:[2,4,1,""],molecule_two:[2,4,1,""],momatrix:[2,4,1,""],multipole:[2,4,1,""],orbital:[2,4,1,""],overlap:[2,4,1,""],projected_atom:[2,4,1,""],unit_atom:[2,4,1,""],visual_atom:[2,4,1,""]},"exatomic.container.Universe":{add_field:[2,3,1,""],add_molecular_orbitals:[2,3,1,""],atom:[2,4,1,""],compute_atom_count:[2,3,1,""],compute_atom_two:[2,3,1,""],compute_bond_count:[2,3,1,""],compute_bonds:[2,3,1,""],compute_density:[2,3,1,""],compute_frame:[2,3,1,""],compute_molecule:[2,3,1,""],compute_molecule_count:[2,3,1,""],compute_unit_atom:[2,3,1,""]},"exatomic.editor":{Editor:[2,1,1,""]},"exatomic.editor.Editor":{parse_frame:[2,3,1,""],to_universe:[2,3,1,""]},"exatomic.error":{AtomicException:[4,6,1,""],BasisSetNotFoundError:[4,6,1,""],ClassificationError:[4,6,1,""],FreeBoundaryUniverseError:[4,6,1,""],PeriodicUniverseError:[4,6,1,""],StringFormulaError:[4,6,1,""]},"exatomic.field":{AtomicField:[14,1,1,""]},"exatomic.field.AtomicField":{compute_dv:[14,3,1,""],integrate:[14,3,1,""],rotate:[14,3,1,""]},"exatomic.filetypes":{cube:[5,0,0,"-"],xyz:[6,0,0,"-"]},"exatomic.filetypes.cube":{Cube:[5,1,1,""],write_cube:[5,5,1,""]},"exatomic.filetypes.cube.Cube":{parse_atom:[5,3,1,""],parse_field:[5,3,1,""]},"exatomic.filetypes.xyz":{XYZ:[6,1,1,""]},"exatomic.filetypes.xyz.XYZ":{from_universe:[6,2,1,""],parse_atom:[6,3,1,""],write:[6,3,1,""]},"exatomic.formula":{SimpleFormula:[2,1,1,""],dict_to_string:[2,5,1,""],string_to_dict:[2,5,1,""]},"exatomic.formula.SimpleFormula":{as_string:[2,3,1,""],mass:[2,4,1,""]},"exatomic.frame":{Frame:[3,1,1,""],compute_frame:[3,5,1,""],compute_frame_from_atom:[3,5,1,""]},"exatomic.frame.Frame":{is_periodic:[3,3,1,""],is_variable_cell:[3,3,1,""]},"exatomic.molecule":{Molecule:[0,1,1,""],compute_molecule:[0,5,1,""],compute_molecule_com:[0,5,1,""],compute_molecule_count:[0,5,1,""]},"exatomic.molecule.Molecule":{classify:[0,3,1,""],get_atom_count:[0,3,1,""],get_formula:[0,3,1,""]},"exatomic.orbital":{DensityMatrix:[14,1,1,""],Excitation:[14,1,1,""],MOMatrix:[14,1,1,""],Orbital:[14,1,1,""]},"exatomic.orbital.DensityMatrix":{from_momatrix:[14,2,1,""],square:[14,3,1,""]},"exatomic.orbital.Excitation":{from_universe:[14,2,1,""]},"exatomic.orbital.MOMatrix":{contributions:[14,3,1,""],square:[14,3,1,""]},"exatomic.orbital.Orbital":{get_orbital:[14,3,1,""]},"exatomic.two":{AtomTwo:[3,1,1,""],MoleculeTwo:[3,1,1,""],compute_atom_two:[3,5,1,""],compute_bond_count:[3,5,1,""],compute_free_two_si:[3,5,1,""],compute_molecule_two:[3,5,1,""],compute_periodic_two_si:[3,5,1,""]},"exatomic.two.AtomTwo":{compute_bonds:[3,3,1,""]},"exatomic.widget":{UniverseWidget:[2,1,1,""]},exatomic:{__init__:[18,0,0,"-"],_config:[4,0,0,"-"],atom:[3,0,0,"-"],basis:[14,0,0,"-"],container:[2,0,0,"-"],editor:[2,0,0,"-"],error:[4,0,0,"-"],field:[14,0,0,"-"],formula:[2,0,0,"-"],four:[0,0,0,"-"],frame:[3,0,0,"-"],molecule:[0,0,0,"-"],orbital:[14,0,0,"-"],three:[0,0,0,"-"],two:[3,0,0,"-"],widget:[2,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","classmethod","Python class method"],"3":["py","method","Python method"],"4":["py","attribute","Python attribute"],"5":["py","function","Python function"],"6":["py","exception","Python exception"]},objtypes:{"0":"py:module","1":"py:class","2":"py:classmethod","3":"py:method","4":"py:attribute","5":"py:function","6":"py:exception"},terms:{"3x3x3":3,"boolean":3,"case":[5,10],"class":[0,2,3,5,6,10,14,18],"default":[0,2,3,6,8,14],"final":14,"float":[2,3,6,14],"function":[0,1,2,3,5,8,10,14,16,18],"int":[2,3,5,14],"return":[0,2,3,6,14],"true":[0,3,6,8,14],For:[2,3,4,14],MOs:10,One:6,The:[1,3,4,5,6,10,14,16],There:14,These:3,about:[2,3,5,14],abov:5,absorb:14,access:14,add:[2,4,8],add_classif:4,add_field:2,add_molecular_orbit:2,addit:[2,3],alia:2,alias:2,all:[2,3,5,10,14,18],along:3,alpha:14,alphabet:2,alreadi:14,also:[5,14],although:18,alwai:5,ambigu:3,analyt:14,analyz:18,angl:14,angstrom:[3,6],angular:14,ani:[0,2,3,14],anoth:3,anyth:3,api:16,appear:18,appli:0,applic:[1,3,13,14],appropri:10,approxim:14,arbitrari:[0,2],arg:[0,2,3,5,8,14],argument:5,around:10,arrai:[3,14],as_intern:[2,6],as_map:0,as_str:2,ask:4,associ:[2,3,14],assum:10,atom0:3,atom1:3,atom:[0,1,5,6,8,16,18],atom_count:3,atom_t:6,atom_two:[2,3],atomiceditor:2,atomicexcept:4,atomicfield:[1,2,5,10,14],atomist:2,atomtwo:[2,3],attach:2,attribut:2,avail:[3,8],awar:5,axi:[2,3],backend:13,base:[1,2,3],basi:[1,2,10,16],basis_funct:14,basis_function_contribut:2,basis_set_ord:2,basissetnotfounderror:4,basissetord:[2,14],becaus:2,below:[0,2,3,14],bent:3,beta:14,between:[2,3,14],bodi:[1,2,16],bohr:3,bond:[2,3,8],bond_extra:[2,3],bool:[3,6,8],both:[3,5],bound:14,boundari:[3,4],builder:3,calcul:[3,14],call:[2,14,18],can:[2,3,14],canon:14,cartesian:3,cat:14,categori:[0,3,14],cell:[3,8],center:[0,3,14],central:3,chang:13,characterist:14,check:[3,14],chemic:[2,14],chemist:16,chemistri:[2,3,14,18],chi0:14,chi1:14,chi2:14,chi:14,classic:2,classif:0,classifi:[0,4],classificationerror:4,classmethod:[3,6,14],cluster:0,code:[4,10,14],coef:[2,14],coeffici:[2,10,14],collect:[3,14],column:[2,3,14],columnar:14,combin:[10,14],come:[2,18],comment:3,common:[14,18],commonli:2,complet:4,comput:[0,2,3,10,14,18],compute_atom_count:2,compute_atom_two:[2,3],compute_bond:[2,3],compute_bond_count:[2,3],compute_dens:2,compute_dv:14,compute_fram:[2,3],compute_frame_from_atom:3,compute_free_two_si:3,compute_molecul:[0,2],compute_molecule_com:0,compute_molecule_count:[0,2],compute_molecule_two:3,compute_nearest_molecul:0,compute_periodic_two:3,compute_periodic_two_si:3,compute_unit_atom:2,concat:2,concaten:3,concept:3,condit:[3,4],configur:[1,16],construct:14,construct_mo:1,contain:[1,2,3,5,8,14,16],context:3,contract:14,contribut:[2,14],contributor:16,control:8,conveni:14,convert:[2,5],convolv:14,coordin:[2,3,6,14],correl:2,correspond:[3,5,6,14],count:[2,3],coval:3,creat:[0,2,6,8,13,14],criteria:14,criterion:14,csv:14,cube:[1,9,14,16],current:[3,6,8,14],custom:[2,8],data:[1,2,5,8,10,14,16,18],datafram:[1,2,3,5,16],defin:[2,8,14],definit:2,degeneraci:14,densiti:[2,14],densitymatrix:[2,14],depend:14,describ:[4,14],descript:[0,2,3,5,6,14],design:14,desir:[5,14],determin:[2,3,14],dict:[2,3],dict_to_str:2,dictionari:[2,3],differ:[3,10],dimens:[9,14],dipol:14,direct:3,directli:[2,14],directori:[5,6],disambigu:14,discret:[5,14],disk:6,displac:3,displai:5,distanc:[2,3],distinguish:3,ditto:3,doe:14,doesn:3,domain:3,due:14,dure:3,dynam:[2,3,4],each:[3,5,14],edit:6,editor:[1,5,16],eigenfunct:14,eigenvalu:14,eigenvector:14,either:14,electron:3,element:[2,3,14],encod:[2,6],energi:[2,14],enumer:[3,14],error:4,essenti:14,establish:14,etc:[0,2,3,14],everyth:2,exa:4,exact:0,exampl:[0,3],exatom:[0,1,2,3,5,6,14],except:[1,16],exchang:2,excit:[2,14],exist:2,experi:2,expon:14,extens:1,extra:[2,3],fact:2,factor:[2,3],fals:[0,2,3,6],fdict:2,featur:2,field3d:14,field:[1,2,5,8,16],field_param:2,field_typ:5,field_valu:[5,14],file:[1,2,9,14,16],filetyp:[1,5,6],filter:2,finit:14,first:[3,5,14],float_format:6,fock:14,follow:[3,14],forc:[2,3],foreign:3,form:[0,14],format:[2,3,6,10,14,18],formula:[0,1,4,16],found:[2,3],four:[1,16],frame:[1,2,5,6,8,14,16],free:[3,4],freeboundaryuniverseerror:4,freqdx:3,frequenc:[2,3],from:[2,3,5,6,8,10,14,18],from_column:14,from_momatrix:14,from_small_molecule_data:3,from_univers:[3,6,14],frontend:13,fstr:2,full:10,fulli:2,gaussian:[1,14,16],gaussianbasisset:14,gaussianorbit:1,gener:[2,3,5,10,14,16],geometri:[2,3,5,8,14],get:[0,2],get_atom_count:0,get_atom_label:3,get_element_mass:3,get_formula:0,get_orbit:14,given:[3,5,6,8,14],going:14,gramian:14,greedi:0,group:14,gtf:1,guess:3,guid:[3,16],hand:14,handl:5,harmon:[1,16],has:[0,4,5,6],have:[3,14],header:3,helper:8,homo:14,homogen:3,how:3,html:8,identifi:[0,3],imag:2,includ:[2,3,14],incorrectli:4,index:[2,3,5,8,14,16],indic:[3,5],individu:[6,14],inform:[2,3,5,14],initi:14,inner:14,instal:16,instead:0,integ:[3,14],integr:14,inter:3,interatom:[2,3],interest:2,irreduc:14,irrep:14,is_period:3,is_variable_cel:3,isotop:3,iter:3,its:14,itself:14,javascript:8,join:2,just:[3,14,18],kei:3,knowledg:10,known:[2,14],kwarg:[0,2,3,5,14],label:[0,3,5,14],laid:10,last:3,last_fram:3,latter:0,len:14,length:[5,6,14],level:3,licens:16,ligand:3,like:[3,14],line:3,linear:[3,10,14],list:[2,3,5,10,14],listen:13,load:[2,8],loop:14,lumo:14,machineri:10,magnet:14,mai:[2,10],major:2,make:14,manipul:14,mapper:[2,3],mass:[0,2,3],math:14,matrix:14,max:[2,14],mean:[3,14],mechan:14,memori:3,mere:10,meta:[2,6],method:14,middl:14,min:2,minim:[2,3],minimum:2,minmal:[2,3],miss:3,mocoef:[2,14],modifi:0,modul:[2,3,4,8,14,16],molecul:[1,2,3,16],molecular:[2,3,14],molecule_count:3,molecule_two:2,moleculetwo:[2,3],momatrix:[2,14],moment:14,momentum:14,more:[6,14],most:[14,18],msg:4,multipl:[3,5],multipol:2,multpl:5,must:14,myfield:14,n_basis_funct:14,name:[2,3,6,18],naoh:2,ndarrai:3,necessari:10,necessarili:10,need:[5,10],neg:14,neglig:14,nest:10,newfield:14,nframe:3,non:[3,14],none:[2,3,5,6,14],normal:14,note:[3,14],notebook:[1,8,16],noth:14,now:5,nprint:[2,6],nstep:2,nuclear:[3,14],number:[0,2,3,14],numer:2,object:[2,3,5],occ:14,occsym:14,occup:[2,14],occupi:14,occvec:14,offset:3,one:[3,6,14],onli:[4,5,14],oper:4,optim:[2,3],option:[4,14],orb:14,orbit:[1,2,8,16],order:[0,2,10,14],order_gtf_basi:1,organ:[2,18],origin:3,osc:14,oscil:[3,14],other:3,out:[10,14],outer:[5,14],outsid:3,over:14,overlap:[0,2,14],overview:16,packag:[2,18],page:16,pair:3,paper:14,parallelpip:14,paramet:[0,2,3,4,5,6,14],parametr:14,pars:[2,5,6],parse_:2,parse_atom:[5,6],parse_field:5,parse_fram:2,part:0,particular:2,pass:[0,5],path:[5,6],path_stream_or_str:[2,6],pattern:14,per:[0,2,5,8],perform:4,period:[2,3,4],periodicuniverseerror:4,phenomena:2,photoelectron:14,physic:5,piec:14,place:[0,3,5],plane:[3,14],platform:16,point:[3,5,6,8],popul:14,posit:[1,14,16],power:14,prefac:14,prefactor:14,present:14,previous:5,primari:3,primit:14,problem:14,program:18,programmat:6,project:3,projected_atom:2,projectedatom:[2,3],properli:10,properti:[1,2,14,16],provid:[2,3,5,14,18],pull:5,put:3,quantiti:[5,18],quantum:[2,14],radii:[2,3],rais:4,rather:8,record:14,refer:5,relat:3,remain:0,remov:2,render:[5,8],repr:[9,14],repres:[2,10],represent:[0,1,2,4,8,10,16],req:[3,14],request:8,requir:[3,5,10],result:[3,14],ret:[3,14],retriev:8,rotat:14,row:14,same:[5,14],scalar:[5,14],scheme:14,search:16,second:14,see:[0,2,3,4,14],select:8,separ:14,seri:[2,3],serial:3,set:[0,1,8,13,16],shape:[5,14],shell:14,shell_funct:14,shorten:18,should:[2,5,14],similar:14,simpl:[1,3,16],simpleformula:[2,4],simplest:3,simul:[2,3,4],slater:14,slice:14,small:3,smallest:3,softwar:18,solid:[1,16],solut:0,solv:14,solvent:0,some:[8,10,18],sourc:[0,2,3,4,5,6,14],space:[5,14],spars:3,spatial:5,specif:[3,4,5,14],specifi:[2,3],spectra:14,spectroscopi:14,sphere:8,spheric:14,spin:14,squar:14,square_planar:3,step:[2,3],sto:14,storag:14,store:[2,14],str:[2,3,5,6,14],strength:14,string:[0,2,4,10],string_to_dict:2,stringformulaerror:4,strongli:2,structur:[2,10,14],subclass:2,sublist:5,success:6,sum:14,sum_:14,summar:14,supercel:3,support:[1,3,14,16],sure:5,symbol:[2,3,14],symmetr:14,symmetri:14,syntax:4,system:[2,3,18],systemat:2,tabl:[1,2,3,6,14,16],tag:3,take:10,test:2,text:2,textbook:14,than:[6,8],them:[2,10],theori:[2,3],therefor:10,thi:[0,2,3,4,5,8,10,14,18],thin:10,those:[0,5],three:[1,16],time:[2,3,14],to_univers:[2,5],to_xyz:3,togeth:2,tol:[2,14],total:3,track:5,trait:5,trajectori:[3,6],transform:[2,14],transit:14,transpar:14,triangular:14,tupl:[0,2],two:[1,2,16],type:[1,2,3,14,16],typic:[3,14],unclassifi:0,uni:[3,5,14],unifi:[2,18],uniqu:[3,14],unique_atom:3,unit:[2,3,5,6,8,18],unit_atom:2,unitari:14,unitatom:[2,3],univers:[0,1,3,4,5,6,14,16],universeapp:1,universetestapp:13,universeview:1,universewidget:2,unknown:14,unoccupi:14,unpack:10,unviers:5,updat:[2,3,8],update_orbit:1,usag:4,use:[2,3,5,14],used:[0,2,3,4,14],user:8,uses:18,using:8,usual:14,uuo:2,valu:[2,5,9,14],vari:3,variabl:[3,8],variou:[3,14],vector:[2,14],veloc:3,via:14,view:[1,8],virt:14,virtsym:14,virtual:14,visual:[1,6,14,16,18],visual_atom:2,visualatom:[2,3],volum:14,wai:[2,10],water:2,wave:14,weight:[10,14],well:[2,5],were:0,when:[2,3,4,8],where:[0,5],whether:2,which:[2,5,14],whose:14,widget:[1,8,16],within:14,without:3,work:10,would:[0,14],wrapper:10,write:[5,6,14],write_cub:5,written:[5,6,14],xyz:[1,3,16],yet:2,your:5,zero:[2,14],zeroth:14},titles:["Three Body Properties Table","API","Atomic Editor","Frame Data","Exatomic Configuration","Cube File Support","XYZ File Editor","Atomic Orbitals","Universe Visualization","Atomic Fields","Gaussian Orbitals","Gaussian Type Functions","Solid Harmonics","Universe Container","Atomic Field","Overview","Exatomic: Unified Computational Chemistry","Installation","Exatomic"],titleterms:{"function":11,The:2,api:1,applic:8,atom:[2,3,7,9,14],atomicfield:9,basi:14,bodi:[0,3],chemistri:16,comput:16,configur:4,construct_mo:10,contain:13,create_gui:8,cube:5,data:3,datafram:14,editor:[2,6],exatom:[4,16,18],except:4,field:[9,14],file:[5,6],formula:2,four:0,frame:3,gaussian:[10,11],gaussianorbit:10,gtf:11,harmon:12,if_empti:13,info:16,init_listen:13,init_var:8,instal:17,molecul:0,notebook:2,orbit:[7,10,14],order_gtf_basi:10,overview:15,posit:3,properti:[0,3],render_cel:8,render_current_fram:8,render_field:8,represent:14,set:14,simpl:2,solid:12,support:5,tabl:0,three:0,two:3,type:11,unifi:16,univers:[2,8,13],universeapp:8,universeview:13,update_field:8,update_orbit:8,view:13,visual:8,widget:2,xyz:6}})