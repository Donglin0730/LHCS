%% LHCS
function[fmin,bestnest,curve,FE]=LHCS(n,dim,fobj)
n_min=5;
n_max=n;
pa_max=0.35;
pa_min=0.25;
pa = pa_min + (pa_max - pa_min)*(1/dim);
FE=0;
MaxFE=dim*10000;
%% Set maximum number of iterations
t_max=0;
if dim==10
t_max=1577;
end
if dim==30
    t_max=4734;
end
if dim==50
    t_max=7891;
end

Lb=-100*ones(1,dim); 
Ub=100*ones(1,dim);
curve=zeros(1,500);
%% Population initialization
for i=1:n
    nest(i,:)=Lb+(Ub-Lb).*rand(size(Lb));
    fitness(i)=fobj(nest(i,:));
end
[fmin,K]=min(fitness);
bestnest=nest(K,:);
[~,index]=sort(fitness);
nestX=zeros(3,dim);
nesta=nest(index(1),:);
nestb=nest(index(2),:);
nestc=nest(index(3),:);
nestX(1,:)=nesta;
nestX(2,:)=nestb;
nestX(3,:)=nestc;

t=0;
while  FE<MaxFE
    t=t+1;
    %%Linear Decreasing Rule Reconciliation Strategies
    if t/t_max<rand
        new_nest=get_cuckoos(n,nest,bestnest,Lb,Ub,t,t_max);   
    else
         nestx=nestX(1,:);
        X1=gwo(n,nest,nestx,t,t_max,Lb,Ub);
        
        nestx=nestX(2,:);
        X2=gwo(n,nest,nestx,t,t_max,Lb,Ub);
        
        nestx=nestX(3,:);
        X3=gwo(n,nest,nestx,t,t_max,Lb,Ub);
        
        new_nest=(X1+X2+X3)/3;
    end
    [fnew,best,nest,fitness]=get_best_nest(n,nest,new_nest,fitness,fobj);
    FE=FE+n;
    if (1-t/t_max)<0.05
        nest=mirror_reflect_learning(n,nest,Lb,Ub);
    end
    new_nest=empty_nests1(n,nest,Lb,Ub,pa) ;
    [fnew,best,nest,fitness]=get_best_nest(n,nest,new_nest,fitness,fobj);
    FE=FE+n;

      [~,index]=sort(fitness);
      nesta=nest(index(1),:);
      nestb=nest(index(2),:);
      nestc=nest(index(3),:);
      nestX(1,:)=nesta;
      nestX(2,:)=nestb;
      nestX(3,:)=nestc;

    %% Linear decreasing strategy for populations
      pn=round((((n_min-n_max)/MaxFE)*FE)+n_max);
      if pn<n
          rn=n-pn;
          if n-rn<n_min
             rn =n-n_min;
          end
          n=n-rn;
          for r=1:rn
              [sortf,index]=sort(fitness,'ascend');
              nest(index(end),:)=[];
              fitness(index(end))=[];
          end
      end 
    if fnew<fmin 
        fmin=fnew; 
        bestnest=best; 
    end
    curve(t)=fmin;
end
end

%% ---------------List of subfunctions------------------
%% Lévy flight Random Walk
function nest=get_cuckoos(n,nest,best,Lb,Ub,t,t_max)
beta=3/2;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);

for j=1:n
    s=nest(j,:);
    %% Lévy flight
    u=randn(size(s))*sigma;
    v=randn(size(s));
    step=u./abs(v).^(1/beta);
    a=0.5*cos((pi/3)*(1+t/(2*t_max)));
    stepsize=a*step.*(s-best);
    s=s+stepsize.*randn(size(s));
   nest(j,:)=simplebounds(s,Lb,Ub);
end
end

%% Hybrid with Gray Wolf Optimization Algorithm
function new_nest=gwo(n,nest,nestx,t,t_max,Lb,Ub)
a=2-t*((2)/t_max);
    r1=rand;
    r2=rand;
    A=2*a*r1-a;
    C=2*r2;
for i=1:n
    s=nest(i,:);
    s=nestx-A.*(C.*nestx-s);
    new_nest(i,:)=simplebounds(s,Lb,Ub);
end
end

%% Incorporating specular reflection learning strategies
function new_nest=mirror_reflect_learning(n,nest,Lb,Ub)
R=rand(1,4);
u=R(1);
Q=R(2);
r4=R(3);
r5=R(4);
if r4>r5
    w=1+u*Q;
else
    w=1-u*Q;
end
for i=1:n
    s=nest(i,:);
    s=(0.5*w+0.5).*(Lb+Ub)-w.*s;
    new_nest(i,:)=simplebounds(s,Lb,Ub);
end
end

%% Find the best nest
function [fmin,best,nest,fitness]=get_best_nest(n,nest,newnest,fitness,fobj)
for j=1:n
    fnew=fobj(newnest(j,:));
    if fnew<=fitness(j)
       fitness(j)=fnew;
       nest(j,:)=newnest(j,:);
    end
end
[fmin,K]=min(fitness) ;
best=nest(K,:);
end

%% Abandon nest, replace old solutions with new ones
function new_nest=empty_nests1(n,nest,Lb,Ub,pa)
K=rand(size(nest))>pa;

stepsize=rand*(nest(randperm(n),:)-nest(randperm(n),:));
new_nest=nest+stepsize.*K;
for j=1:n
    s=new_nest(j,:);
    new_nest(j,:)=simplebounds(s,Lb,Ub);
end
end

%% Simple boundary
function s=simplebounds(s,Lb,Ub)
  ns_tmp=s;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);

  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  
  s=ns_tmp;
end