## Guidance Graph Optimization for Lifelong Multi-Agent Path Finding

## Yulun Zhang^1 , He Jiang^1 ,Varun Bhatt^2 , Stefanos Nikolaidis^2 and Jiaoyang Li^1

(^1) Robotics Institute, Carnegie Mellon University
(^2) Thomas Lord Department of Computer Science, University of Southern California

## {yulunzhang,hejiangrivers}@cmu.edu,{vsbhatt,nikolaid}@usc.edu, jiaoyangli@cmu.edu

## Abstract

```
We study how to use guidance to improve the
throughput of lifelong Multi-Agent Path Finding
(MAPF). Previous studies have demonstrated that,
while incorporating guidance, such as highways,
can accelerate MAPF algorithms, this often results
in a trade-off with solution quality. In addition,
how to generate good guidance automatically
remains largely unexplored, with current methods
falling short of surpassing manually designed ones.
In this work, we introduce the guidance graph as
a versatile representation of guidance for lifelong
MAPF, framing Guidance Graph Optimization as
the task of optimizing its edge weights. We present
two GGO algorithms to automatically generate
guidance for arbitrary lifelong MAPF algorithms
and maps. The first method directly optimizes edge
weights, while the second method optimizes an
update model capable of generating edge weights.
Empirically, we show that (1) our guidance graphs
improve the throughput of three representative life-
long MAPF algorithms in eight benchmark maps,
and (2) our update model can generate guidance
graphs for as large as 93 × 91 maps and as many
as 3,000 agents. We include the source code at:
https://github.com/lunjohnzhang/
ggo_public. All optimized guidance graphs are
available online at: https://yulunzhang.
net/publication/zhang2024ggo.
```

## 1 Introduction

```
We study the problem of leveraging a guidance graph with op-
timized edge weights to guide agent movement, thereby im-
proving the throughput of lifelong Multi-Agent Path Finding
(MAPF). MAPF [Sternet al., 2019] aims to plan collision-
free paths for a set of agents from their start to goal locations
on a given map, depicted as a graphG. Lifelong MAPF [Liet
al., 2021] extends MAPF by assigning new goals to agents as
soon as they reach their current ones. Example applications
include character control in video games [Maet al., 2017b;
Jansen and Sturtevant, 2008] and automated warehouses in
which hundreds of robots are continually assigned new tasks
to transport inventory pods [Varamballyet al., 2022]. Driven
```

```
(a) No guidance (b) Crisscross (c) Our guidance
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
0.
```

```
Figure 1: Comparison of no guidance, human-designed crisscross
guidance, and our guidance with a simulation of 240 agents in a 33
×36 warehouse map, shown in Figure 4b. The heatmaps show the
tile-usage (the frequency that each tile is occupied). Our guidance
results in the most balanced traffic with the highest throughput.
```

```
by these real-world demands, numerous studies have focused
on improving the throughput, namely the average number
of reached goals per timestep, by developing better lifelong
MAPF algorithms [Maet al., 2017a; Liet al., 2020; Kouet
al., 2020; Damaniet al., 2021] or optimizing map layouts
(i.e., map structures) [Zhanget al., 2023a,b].
Given that lifelong MAPF requires online computation of
new paths as agents are continuously assigned to new goal
locations, lifelong MAPF algorithms always decompose the
problem into a series of (one-shot) MAPF instances and solve
them sequentially. However, such methods are myopic be-
cause each MAPF instance involves only the current goal
locations. Achieving (near-)optimal solutions for individ-
ual MAPF instances does not necessarily result in the best
throughput. In this work, we propose to foster implicit co-
operation among agents over the long term by introducing
global guidance for agent movement. Our guidance takes
the form of a directed weighted graph that alters the costs
of agents moving along each edge and waiting at each ver-
tex of graphG. Intuitively, such a guidance graph serves two
purposes. First, by amplifying the cost difference of traveling
through an edge in opposite directions, we encourage agents
to move in the same direction, reducing the number ofhead-
oncollisions, which happen when two agents try to traverse
through the same edge in opposite directions. Second, by in-
creasing the cost of moving in areas prone to congestion, we
motivate agents to navigate through less congested areas, ul-
timately reducing traffic congestion. Figure 1 illustrates the
traffic resulting from different guidance strategies.
```

# arXiv:2402.01446v2 [cs.MA] 9 May 2024

One closely related work to our guidance graph is high-
ways [Cohenet al., 2015; Li and Sun, 2023], which are a
subset of edges selected from graphGwith assigned direc-
tions and a lower traversal cost. This strategy incentivizes
agents to move along the highways, reducing the number of
collisions to be resolved by MAPF algorithms. However, the
question of how to select these edges and determine their di-
rections and costs remains largely unexplored. Thecrisscross
approach [Cohen, 2020], where edge directions alternate in
even and odd rows and columns, is common but not opti-
mal. Li and Sun [2023] show that, while crisscross highways
speed up lifelong MAPF algorithms, they do not always im-
prove throughput. Additionally, Cohenet al.[2016] proposed
two methods to select edges and directions for highways, but
neither of them outperforms crisscross highways.
Therefore, we introduce the Guidance Graph Optimiza-
tion (GGO) problem to improve throughput by optimizing
the edge weights of a guidance graph. We present two au-
tomatic GGO methods. The first applies Covariance Ma-
trix Adaptation Evolutionary Strategy (CMA-ES) [Hansen,
2016], a state-of-the-art black-box optimizer, to solve GGO,
but its solution is map-specific. The second method, Parame-
terized Iterative Update (PIU), uses CMA-ES to optimize an
update model. This update model, represented by a neural
network, starts with an unweighted guidance graph and itera-
tively updates it with traffic information obtained from a life-
long MAPF simulator. It is capable of optimizing guidance
graphs for different maps with similar layouts.
We make the following contributions: (1) introducing the
guidance graph, a versatile representation of guidance for
lifelong MAPF, and guidance graph optimization (GGO) to
improve its throughput, (2) conducting an in-depth study of
various existing guidance works in MAPF, and (3) proposing
two automatic GGO methods, CMA-ES and PIU, showcas-
ing their superior performance over unweighted graphs and
previous guidance methods, along with the transferability of
PIU to larger maps with similar layouts.

## 2 Problem Definition and Preliminaries

### 2.1 Lifelong Multi-Agent Path Finding

Definition 1(MAPF). The (one-shot) MAPF problem takes
as inputs a graphG(V,E)andkagents with their start and
goal locations. At each timestep, an agent can move to an ad-
jacent vertex or stay at its current vertex. Two agents collide
when they arrive at the same vertex or swap locations at the
same timestep. The MAPF problem searches for collision-
free paths that move all agents from their start to goal loca-
tions with minimum sum-of-cost, defined as the total number
of move and wait actions that the agents need to take.

Definition 2(Lifelong MAPF).Lifelong MAPF extends one-
shot MAPF by constantly assigning new goals to agents when
they reach their current ones. Lifelong MAPF searches for
collision-free paths that maximize throughput, namely the av-
erage number of reached goals per timestep.

Lifelong MAPF Algorithms
Solving MAPF optimally is known to be NP-hard [Yu and
LaValle, 2013]. Lifelong MAPF poses an even greater chal-
lenge as agents consistently receive new goals, requiring the

```
continuous computation of new paths. Consequently, state-
of-the-art algorithms approach lifelong MAPF by decompos-
ing it into a series of (modified) one-shot MAPF instances,
usually one at each timestep, assuming that minimizing their
sum-of-costs enhances lifelong MAPF throughput. They can
be divided into three categories. To show the generality of
our GGO algorithms, we select a leading algorithm from each
category to conduct our experiments.
Replan All.We replan all agents at every timestep (or every
few timesteps) [Wanet al., 2018; Liet al., 2021]. In each
replanning cycle, we solve a MAPF instance with the start lo-
cations being the current locations of all agents and the goals
being their current goals. We select RHCR [Liet al., 2021]
as a representative algorithm from this category.
Replan New.This category is similar to the previous one ex-
cept that, at every timestep, we replan only agents that have
just reached their current goals and have been assigned new
goals [C ́apet al., 2015; Maet al., 2017a; Grenouilleauet al.,
2019; Liuet al., 2019]. Since agents being replanned must
avoid collisions with agents not being replanned, methods in
this category need to impose constraints on the map structure
and goal locations, often denoted as well-formed maps, to en-
sure the existence of collision-free paths. We select Dummy
Path Planning (DPP) [Liuet al., 2019; Liet al., 2021] as a
representative algorithm from this category.
Reactive. In contrast to the previous two categories, reac-
tive methods plan paths for each agent without considering
collisions with other agents (resulting in paths with no wait
actions) and then resolve collisions reactively through pre-
defined rules [Wang and Botea, 2008; Okumuraet al., 2019;
Yu and Wolf, 2023], such as inserting wait actions or taking
short detours. We select PIBT [Okumuraet al., 2019] as a
representative algorithm from this category. It is complete
on biconnected graphs and runs significantly faster than algo-
rithms in other categories.
```

### 2.2 Guidance Graph Optimization

```
To maintain generality, we consider graphG(V,E)to be ei-
ther directed, undirected, or mixed. We use{u,v}and(u,v)
to denote an undirected edge and a directed edge, respec-
tively. We useEundandEdirto denote the subsets of edges
inEthat are undirected and directed, respectively.
Definition 3(Guidance Graph). Given a graphG(V,E)for
lifelong MAPF, we define a guidance graph as a directed
weighted graphGg(Vg,Eg,ω)with the same vertex setVg=
V. Each edge inEgcorresponds to an action that an agent
can take at each vertex, with the edge weight indicating the
action cost. Formally, we defineEg=Ewait∪Emovewith
```

```
Ewait=
```

#### [

```
v∈V
```

```
{(v,v)} (1)
```

```
Emove={
```

#### [

```
{u,v}∈Eund
```

```
{(u,v),(v,u)}}∪Edir. (2)
```

```
All edges weights are collectively represented as a vectorω∈
R
```

```
|Eg|
> 0.
Planning with guidance graphs.To utilize guidance graphs
in lifelong MAPF, we redefine the sum-of-costs of the un-
derlying (one-shot) MAPF instances as the sum of the action
```

```
Representation Generation Usage
Edge direction Move cost Wait cost Design MAPF Online Update Method
1 Jansen and Sturtevant [2008] soft R+ N/A handcrafted procedure lifelong Yes reactive
2 Wang and Botea [2008] strict 1 N/A crisscross one-shot No reactive
3 Cohenet al.[2015] soft 1 orc 1 crisscross one-shot No ECBS
4 Cohenet al.[2016] soft 1 orc 1 handcrafted procedure one-shot No ECBS
5 Han and Yu [2022] soft R+ N/A handcrafted procedure both Yes reactive & RHCR
6 Li and Sun [2023] soft 1 orc 1 crisscross lifelong No RHCR
7 Yu and Wolf [2023] soft R+ N/A handcrafted procedure lifelong Yes reactive
8 Chenet al.[2024] soft R+ N/A handcrafted procedure both Yes PIBT
9 GGO (ours) soft R+ R+ automatic lifelong No many
```

Table 1: Overview of previous works on representation, generation, and usage of guidance in MAPF. For edge direction, strict means
movement is unidirectional along each edge, and soft means bidirectional. Move cost refers to the cost of moving to an adjacent vertex and
wait cost refers to the cost of waiting at the current vertex. For move cost, “1 orc” means that the value considers 1 and a scalarc > 1 only.
For design, handcrafted procedure refers to using a manually designed procedure to generate guidance, while crisscross refers to the popular
human-designed guidance [Cohen, 2020].

costs across all paths for all agents (instead of the total num-
ber of actions). This leads to a minor modification to existing
lifelong MAPF algorithms. Specifically, when planning paths
for each agent, instead of seeking the shortest path onG, we
aim to find a cost-minimal path onGg. This modification
alters the MAPF objective without compromising feasibility.

Definition 4(Guidance Graph Optimization (GGO)).Given
a graphG(V,E), an objective functionf :R|Eg|→R, as
well as predefined lower and upper boundsωlbandωub( 0 <
ωlb≤ωub) for edge weights, the GGO problem searches for
the optimal guidance graphG∗g(Vg,Eg,ω∗)with

```
ω∗= arg max
ωlb≤ω≤ωub
```

```
f(ω). (3)
```

In this paper, our objective functionfis a simulator that
runs a given lifelong MAPF algorithm on a given guidance
graph and returns the throughput.

## 3 Guidance in MAPF

While the term “guidance” has not been explicitly proposed
in the MAPF literature, the concept of enforcing global guid-
ance and rules to enhance MAPF has been employed by nu-
merous works in various ways. In this section, we present a
summary of these works and provide a comprehensive review
of how they represent, generate, and utilize guidance. Table 1
shows a comparison between them and our GGO. We refer to
them by their indices in the table for the rest of this section.
Representing Guidance. Previous works primarily repre-
sent guidance through modified edge directions or move-
ment costs. They are all particular cases within the defini-
tion of our guidance graph, and none of them consider vary-
ing wait costs. We roughly divide them into 4 categories.
(1) Inspired by potential-field and flow-field methods used in
swarm robotics, Work 1 represents guidance through a di-
rection map. This map assigns a direction vector to every
vertex ofGand sets the movement cost along an edge to
be inversely related to the dot product of its direction vec-
tor and the vector of the edge, encouraging agents to move
along the direction vectors. Consequently, a direction map
can be transformed into a guidance graph with edge weights
defined as dot products mentioned above. (2) Work 2 turns

```
some undirected edges into unidirectional, strictly prohibit-
ing agents from moving against the assigned edge directions.
This can be seen as a special guidance graph with infinite
weights for constrained edges and 1 for others. (3) Works 3,
4, and 6 use the highway idea that converts undirected edges
into directed edges in both directions and then selects a subset
of directed edges to be highways. They assign a weight of 1
for all highway edges and a weight of a predefined constant
c > 1 for other edges, encouraging agents to move along
the highway edges. Highways are special guidance graphs
with restrictions on the values of movement and wait costs.
(4) Work 5 uses a temporal heuristic function to estimate the
movement cost between adjacent vertices at each timestep.
This temporal heuristic function can be viewed as a time-
extended guidance graph where edge weights can be different
at different timesteps. However, the time-extended guidance
graph is only applicable while an online update mechanism
is incorporated (see below for details). (5) Works 7 and 8
represent guidance similarly to our guidance graphs, with the
distinction that they do not allow self-edges and thus cannot
represent wait costs.
```

```
Generating Guidance. An important distinction between
our work and previous works is that we are the first to pro-
pose an automated method for generating guidance. All pre-
vious works either use handcrafted guidance, such as criss-
cross highways (Works 2, 3, and 6), or use handcrafted pro-
cedures to generate guidance from a heatmap or a similar data
structure that predicts traffic flows (Works 1, 4, 5, 7, and 8).
More specifically, Work 1 computes direction vectors (of the
direction map) from past traffic flows and then uses a hand-
crafted equation to convert them into movement costs. Work
4 introduces two methods, GM and HM, for generating high-
ways. GM uses a graphic model [Koller and Friedman, 2009]
with a number of handcrafted features obtained from the es-
timated traffic flow. HM converts the estimated traffic flow
into a score for each edge using a handcrafted score func-
tion and selects edges based on a predefined score threshold.
Works 5 and 8 collect the planned paths of all agents and
convert them into movement costs through handcrafted equa-
tions. Last, Work 7 uses a data-driven model to predict traffic
flow, or more specifically, the delays that the agents will en-
counter (due to collision avoidance etc.) and directly uses the
```

```
Sample Form guidance graphRun lifelong MAPF Algo Update
```

Figure 2: CMA-ES for GGO. The edge weights are iteratively sam-
pled from a Gaussian distribution and then evaluated by a lifelong
MAPF simulator. The simulated results are used to update the Gaus-
sian distribution towards high-throughput regions.

predicted delays as movement costs.
Therefore, we select 4 baseline methods to generate guid-
ance graphs in our experiments: (1)Unweighted, where no
guidance is used, (2)Crisscross, (3)HM Cost, adapted from
HM in Work 4, and (4)Traffic Flow, adapted from Work 8.
We did not compare Work 1, as HM from Work 4 is inspired
by it. We did not compare GM from Work 4, as its perfor-
mance is similar to HM. We did not compare Work 5 because
the non-temporal version of their proposed guidance is simi-
lar to Works 4 and 8. We did not compare Work 7, because,
while it obtains predicted traffic flow differently from Work 4,
the procedure of converting predicted traffic flow to guidance
is similar.
Please note that the HM Cost and Traffic Flow used in
our experiments do not use the original traffic flow models
in their papers. This is because Work 4 tackles one-shot
MAPF and predicts traffic flows by planning shortest paths
between the start and goal locations of the agents, which is
not realistic in lifelong MAPF as goal locations are unknown
in advance. Work 8 tackles lifelong MAPF but assumes the
guidance graph can be updated on the fly using real-time traf-
fic information, while we assume that our guidance graph is
optimized offline, and thus we do not have access to real-
time traffic information. We focus on optimizing the guidance
graphs offline because adding online adaptation not only re-
quires additional computation to update the guidance graph
but can also dramatically slow down path planning. This is
because we need to either update the heuristic for the single-
agent A∗search when the guidance graph is updated or use
a less informed heuristic without update. For example, Work
8 incorporates an online adaptation mechanism in PIBT, but
it slows down the algorithm by 2-10 times. Thus, in this pa-
per, we use the same traffic flow model, namely the tile-usage
map obtained from simulation, for both HM Cost and Traffic
Flow methods. More details can be found in Appendix A.
Using Guidance. All previous works study their guid-
ance methods with a specific MAPF algorithm, such as
ECBS [Bareret al., 2014] and RHCR [Liet al., 2021], so
it remains unclear whether and how well their methods can
generalize to other MAPF algorithms. For example, an ev-
ident limitation of methods designed for reactive (lifelong)
MAPF algorithms (namely Works 1, 2, 5, 7, and 8) is that,
since paths planned by reactive methods do not include wait
actions, these guidance methods, by design, do not define
wait costs, making it non-trivial to extend them to (lifelong)
MAPF algorithms in other categories. In contrast, we assess
our GGO methods with three leading lifelong MAPF algo-

```
rithms from different categories, thereby demonstrating their
generality.
```

## 4 Approach

```
We first introduce CMA-ES to solve GGO directly. Then we
introduce Parameterized Iterative Update (PIU), which uses
CMA-ES to optimize an update model that iteratively gener-
ates a guidance graph based on simulated traffic information.
```

### 4.1 CMA-ES

```
CMA-ES [Hansen, 2016] is a derivative-free, black-box op-
timization algorithm based on covariance matrix adaptation.
Figure 2 gives an overview of using CMA-ES to solve GGO.
Specifically, we model the edge weight vectorωas a mul-
tivariate Gaussian distribution. We then iteratively sample
from the distribution for a new batch ofbedge weight vec-
tors, formingbguidance graphs. We normalize eachωto
meet the bound constraintωlb≤ω≤ωub. We then evalu-
ate each guidance graph by runningNecmasimulations in
a given lifelong MAPF simulator and computing the aver-
age throughput. The evaluated guidance graphs are ranked
based on their throughput, and the topNbestof them are used
to update the mean and covariance of the Gaussian distribu-
tion. We run CMA-ES forIiterations and return the guidance
graph with the highest throughput as the solution.
Handling Bounds through Normalization. We use min-
max normalization to enforce the bound constraint because it
does not affect path-planning solutions. To prove it, consider
two guidance graphs with edge weightsω 1 andω 2 =C·ω 1 ,
whereC∈R+. Since the weight of every edge is scaled by
the same scalerC, the paths returned by the lifelong MAPF
algorithms with low-level single agent solvers minimizing the
sum of edge weights do not change. We show additional
experiments in Appendix B.1 that min-max normalization
yields better solutions than representative bounds handling
methods introduced in a prior study Biedrzycki [2020].
```

### 4.2 Parameterized Iterative Update

```
CMA-ES is known to scale poorly to high dimensional search
spaces, making it challenging to optimize guidance graph for
large maps. Therefore, we propose Parameterized Iterative
Update (PIU). Figure 3 gives an overview of PIU, and Al-
gorithm 1 provides the pseudocode. On a high level, PIU
leverages a parameterizedupdate modelto iteratively update
the edge weights of the guidance graph using traffic informa-
tion obtained from lifelong MAPF simulations. PIU can work
with a wide variety of optimization methods. In this work, we
choose to use CMA-ES to optimize the update model.
Definition 5 (Update model). Given a guidance graph
Gg(Vg,Eg,ω), an update model is a functionπθ:R|Eg|×
R|Eg| → R|Eg|that computes the updated edge weights
ω′∈R|>E 0 g|given the current edge weightsω∈R|>E 0 g|and
edge usageUEg ∈R|≥E 0 g|. The edge usage is the frequency
with which each edge is used by the agents in the lifelong
MAPF simulation. The modelπθis parameterized by a vec-
torθ∈Θ, whereΘis the space of all parameters.
```

```
Obtain edge
usage
```

Figure 3: PIU for GGO. Starting with a guidance graph with uniform
edge weights, we run MAPF simulations to get the edge usage. We
then use an update modelπθto update the edge weights. We run
this process iteratively forNpiterations. The update modelπθis
optimized using CMA-ES.

PIU.The red loop in Figure 2 gives an overview of PIU, and
Lines 9 to 19 of Algorithm 1 provides the pseudocode. We
first form an update modelπθparameterized by a given pa-
rameter vectorθ(Line 10). We then start an iterative update
procedure (Lines 11 to 18). In each iteration, we either initial-
ize the edge weights to 1 (Line 12) in the first iteration or use
the update modelπθto update the edge weights (Line 13) in
the following iterations. We then construct the current guid-
ance graph usingω(Line 14). We then runNepiulifelong
MAPF simulations (Lines 15 and 16), computing the average
throughputfand edge usageUEg(Lines 17 and 18). We run
PIU forNpiterations. Finally, we return the throughputfof
the last iteration (Line 19).

Update Model Optimization.To train the update model, we
run the PIU algorithm forNpiterations with update model
πθgiven parametersθ. We search for optimal parameters
θ∗= arg maxθ∈ΘPIU(θ,Np)using CMA-ES. Lines 1 to 8
of Algorithm 1 show the pseudocode. Starting with a given
initial multivariate Gaussian distribution (Line 1), the algo-
rithm samplesbparameter vectors (Line 3) and runs PIU with
them (Lines 4 and 5). Based on the returned throughput val-
ues, it keeps track of the best update model (Line 6) and up-
dates the Gaussian distribution (Line 7), starting a new itera-
tion. The optimization ends after running the above process
forIiterations (Line 2). Finally, the algorithm returns the
best update model and the corresponding throughput (Line 8).

On the Advantage of PIU.Compared to directly using
CMA-ES, the advantage of optimizing the update model and
using PIU to generate guidance graph is two-folds. First, op-
timizing the update model reduces the dimension of search
space. Although solving GGO directly with CMA-ES is ver-
satile and applicable to various lifelong MAPF algorithms
and maps, its effectiveness diminishes in high-dimensional
search spaces, making it challenging to use CMA-ES to
search for edge weights directly for large maps. Specifically,

```
Algorithm 1:Update model optimization
Input: μ 0 ,Σ 0 : initial mean and covariance matrix of
the multivariate Gaussian distribution.
Np: number of iterations to run inPIU.
updategaussian: function to updateμandΣ
according to evaluated parameter vectors.
simulate: function to run lifelong MAPF
simulation and return edge usageUEgand
throughputf.
1 Initializeμ←μ 0 ,Σ←Σ 0 ,θ∗←NULL,g∗←−∞
2 fori← 1 toIdo
3 Samplebparameter vectorsθ 1 ,...,θb∼N(μ,Σ)
4 fork← 1 tobdo
5 gk←PIU(θk,Np)
6 ifgk> g∗then g∗←gk,θ∗←θk
7 μ,Σ←updategaussian(g 1 ∼b,θ 1 ∼b)
8 returng∗,θ∗
9 FunctionPIU(θ,Np):
10 Construct update modelπθ
11 forj← 1 toNpdo
12 ifj= 1then ω← 1
13 else ω←πθ(ω,UEg)
14 Construct guidance graphGg(Vg,Eg,ω)
15 forq← 1 toNepiudo
16 UE(qg),f(q)←simulate(Gg)
```

```
17 UEg←Ne^1 piu
```

```
PNepiu
q=1 U
```

```
(q)
Eg
18 f←Ne^1 piu
```

```
PNepiu
q=1 f
```

```
(q)
```

```
19 returnf
```

```
CMA-ES employs a full-rankn×ncovariance matrix to
model its Gaussian distribution in ann-dimensional space,
leading to quadratic increases in both time and space com-
plexity [Varelaset al., 2018]. In the case of GGO, the num-
ber of edge weights of a guidance graph increases at least
linearly with the number of vertices, while our update model
maintains a consistent number of parameters regardless of the
size of the guidance graph, offering a more scalable solution
than directly applying CMA-ES. Second, the optimized up-
date model is not specific to the map it is optimized on. Dif-
ferent maps with similar layouts could potentially have sim-
ilar high-throughput guidance graphs that can be generated
by the same update model. The guidance graph optimized by
CMA-ES, on the other hand, consists of edge weights for a
specific map.
```

## 5 Experimental Evaluation

```
In this section, we compare guidance graphs optimized by
CMA-ES and PIU with various baselines and assess the ca-
pability of PIU to generate high-throughput guidance graphs
for maps of larger sizes with similar layouts.
```

### 5.1 Experiment Setup

```
General Setups. Table 2 outlines our experimental setup.
Column 2 shows the lifelong MAPF algorithms. Follow-
```

```
PIBT + CMA-ES PIBT + PIU
RHCR + CMA-ES
```

```
DPP + CMA-ES
PIBT + Crisscross
RHCR + Crisscross
```

```
DPP + Crisscross
PIBT + HM Cost
RHCR + HM Cost
```

```
DPP + HM Cost
PIBT + Traffic Flow
RHCR + Traffic Flow
```

```
DPP + Traffic Flow
PIBT + Unweighted
RHCR + Unweighted
```

```
DPP + Unweighted
```

```
200 400 800
Number of Agents
```

```
0
```

```
4.
```

```
9
```

```
Throughput
```

```
(a) Setup 1:random-32-32-
```

```
50 220 400 800
Number of Agents
```

```
0
```

```
4
```

```
8
```

```
Throughput
```

```
(b) Setup 2 & 8:warehouse-33-
```

```
100 400 700
Number of Agents
```

```
0
```

```
0.
```

```
1.
```

```
Throughput
```

```
(c) Setup 3:maze-32-32-
```

```
200 1000 1800
Number of Agents
```

```
0
```

```
16
```

```
32
```

```
Throughput
```

```
(d) Setup 4:empty-48-
```

```
100 1500 3000
Number of Agents
```

```
0
```

```
1.
```

```
3.
```

```
Throughput
```

```
(e) Setup 5:room-64-64-
```

```
500 1500 3000
Number of Agents
```

```
0
```

```
5
```

```
10
```

```
Throughput
```

```
(f) Setup 6:random-64-64-
```

```
200 1200 2000
Number of Agents
```

```
0
```

```
2.
```

```
5
```

```
Throughput
```

```
(g) Setup 7:den312d
```

```
10 49 8888
Number of Agents
```

```
0
```

```
3
```

```
6
```

```
Throughput
```

```
(h) Setup 9:warehouse-20-
```

Figure 4: Throughput with different numbers of agents. The guidance graphs are optimized withNaagents, which is indicated by the black
vertical lines. In (b), the black vertical lines at 220 and 400 agents indicateNafor setups 8 and 2, respectively.

```
Setup MAPF Map |Eg| |Ewait| |Emove| Na GGO
1
```

```
PIBT
```

```
random-32-32-20 3,359 819 2,
400
CMA-ES & PIU
```

```
2 warehouse-33-36 4,074 948 3,
3 maze-32-32-4 3,484 790 2,
4 empty-48-48 11,328 2,304 9,024 1,
5 room-64-64-8 14,340 3,232 11,108 1,
6 random-64-64-20 13,568 3,270 10,
7 den312d 11,227 2,445 8,782 1,
8 RHCR warehouse-33-36 4,074 948 3,126 220 CMA-ES
9 DPP warehouse-20-17 1,478 320 1,158 88 CMA-ES
10 PIBT warehouse-33-36 4,074 948 3,126 150 CMA-ES & PIU
```

Table 2: Summary of the experiment setup. Nais the number of
agents.|Eg|,|Ewait|, and|Emove|are the number of wait edges,
movement edges, and all edges in the guidance graph, respectively.
Setups 1 to 9 compare our optimized guidance graphs with the base-
lines. Setup 10 compares PIBT with GGO against RHCR without
GGO when there are fewer agents.

ing the recommendations of previous works [Liet al., 2021;
Zhanget al., 2023b], we use PBS [Maet al., 2019] and
SIPP [Phillips and Likhachev, 2011] as the MAPF solver and
the single-agent solver, respectively, in both RHCR and DPP
and useh= 5andw= 10in RHCR.
Column 3 outlines the maps, all being 4-neighbor
girds, including two warehouse maps (warehouse-33-36and
warehouse-20-17) from previous works [Liet al., 2021;
Zhanget al., 2023b] and six maps (random-32-32-20,maze-
32-32-4,empty-48-48,room-64-64-8,random-64-64-20, and
den312d) from the MAPF benchmark [Sternet al., 2019].
We choose multiple maps for PIBT to demonstrate that both
CMA-ES and PIU work for different maps. We show the

```
maps at the corners of Figure 4. In all maps, black tiles
are obstacles, and non-black tiles are traversable. In ware-
house maps, orange tiles are home locations, blue tiles are
endpoints, and purple tiles are workstations. Inwarehouse-
20-17, agents start from orange tiles and move constantly be-
tween blue tiles. Inwarehouse-33-36, agents start from non-
black tiles and move between blue and purple tiles alterna-
tively. In all other maps, agents start from and move between
white tiles. The agents’ goals are assigned uniform randomly.
Columns 4 to 6 show the number of edges in the guidance
graphs of the corresponding maps. In setups 1 to 7, we opti-
mize all|Eg|=|Ewait|+|Emove|edges. In setups 8 and 9,
however, SIPP cannot handle different wait costs at different
vertices. Therefore, we optimize the wait costs of all vertices
as one variable, resulting in|Emove|+ 1 = 3, 127 and 1 , 159
variables to be optimized in the guidance graphs in setups 8
and 9, respectively. Column 7 is the number of agents used in
lifelong MAPF simulations, with a largerNafor PIBT com-
pared to RHCR and DPP to demonstrate that both CMA-ES
and PIU work for congested scenarios.
Column 8 shows the GGO algorithms we run for each
setup. We apply CMA-ES across all setups to demonstrate
its versatility. However, due to computational constraints, we
focus on using PIU primarily with PIBT. While both PIU and
CMA-ES conduct the same number of simulations, there is a
notable difference in their execution. In CMA-ES, allNecma
simulations in each guidance graph evaluation can be paral-
lelized. In contrast, PIU runs the simulations sequentially,
resulting in slower runtime.
We choose the hyperparameters of CMA-ES and PIU such
```

```
Setup MAPF + GGO SR Throughput CPU Runtime (s)
```

```
1
```

```
PIBT + CMA-ES 100% 7.78±0.02 1. 31 ± 0. 01
PIBT + PIU 100% 7. 46 ± 0. 02 1. 29 ± 0. 02
PIBT + Crisscross 100% 6. 84 ± 0. 02 1. 24 ± 0. 02
PIBT + HM Cost 100% 5. 98 ± 0. 02 1.17±0.
PIBT + Traffic Flow 100% 7. 43 ± 0. 02 1. 19 ± 0. 02
PIBT + Unweighted 100% 5. 52 ± 0. 01 1. 20 ± 0. 02
```

```
2
```

```
PIBT + CMA-ES 100% 7.64±0.01 1. 27 ± 0. 01
PIBT + PIU 100% 7. 28 ± 0. 01 1. 22 ± 0. 01
PIBT + Crisscross 100% 6. 65 ± 0. 01 1.21±0.
PIBT + HM Cost 100% 5. 63 ± 0. 01 1. 24 ± 0. 01
PIBT + Traffic Flow 100% 5. 84 ± 0. 01 1. 23 ± 0. 01
PIBT + Unweighted 100% 5. 22 ± 0. 01 1. 25 ± 0. 01
```

```
3
```

```
PIBT + CMA-ES 100% 1. 40 ± 0. 03 0. 60 ± 0. 01
PIBT + PIU 100% 1.47±0.02 0. 60 ± 0. 01
PIBT + Crisscross 100% 1. 18 ± 0. 03 0. 59 ± 0. 01
PIBT + HM Cost 100% 1. 16 ± 0. 02 0. 65 ± 0. 01
PIBT + Traffic Flow 100% 0. 95 ± 0. 02 0. 63 ± 0. 01
PIBT + Unweighted 100% 1. 09 ± 0. 02 0.58±0.
```

```
4
```

```
PIBT + CMA-ES 100% 24. 04 ± 0. 01 2. 86 ± 0. 05
PIBT + PIU 100% 25.98±0.01 2.28±0.
PIBT + Crisscross 100% 23. 84 ± 0. 02 2. 82 ± 0. 04
PIBT + HM Cost 100% 20. 90 ± 0. 02 2. 79 ± 0. 05
PIBT + Traffic Flow 100% 19. 90 ± 0. 02 2. 73 ± 0. 04
PIBT + Unweighted 100% 19. 48 ± 0. 03 2. 78 ± 0. 04
```

```
5
```

```
PIBT + CMA-ES 100% 3. 12 ± 0. 01 4. 91 ± 0. 06
PIBT + PIU 100% 3.13±0.01 4. 59 ± 0. 05
PIBT + Crisscross 100% 2. 75 ± 0. 01 4. 67 ± 0. 07
PIBT + HM Cost 100% 2. 41 ± 0. 01 4. 71 ± 0. 05
PIBT + Traffic Flow 100% 2. 87 ± 0. 01 4.55±0.
PIBT + Unweighted 100% 2. 51 ± 0. 01 4. 95 ± 0. 06
```

```
6
```

```
PIBT + CMA-ES 100% 9.00±0.07 3. 52 ± 0. 06
PIBT + PIU 100% 8. 43 ± 0. 11 4. 25 ± 0. 07
PIBT + Crisscross 100% 7. 31 ± 0. 09 3. 55 ± 0. 05
PIBT + HM Cost 100% 6. 45 ± 0. 10 3. 75 ± 0. 06
PIBT + Traffic Flow 100% 6. 00 ± 0. 12 3.39±0.
PIBT + Unweighted 100% 6. 01 ± 0. 09 3. 45 ± 0. 05
```

```
7
```

```
PIBT + CMA-ES 100% 4.98±0.01 2. 71 ± 0. 04
PIBT + PIU 100% 4. 87 ± 0. 01 2.61±0.
PIBT + Crisscross 100% 4. 16 ± 0. 01 3. 27 ± 0. 05
PIBT + HM Cost 100% 3. 99 ± 0. 02 2. 90 ± 0. 06
PIBT + Traffic Flow 100% 3. 06 ± 0. 01 2. 73 ± 0. 06
PIBT + Unweighted 100% 3. 05 ± 0. 01 3. 17 ± 0. 05
```

```
8
```

```
RHCR + CMA-ES 100% 6.58±0.04 91.24±15.
RHCR + Crisscross 100% 5. 59 ± 0. 20 3771. 71 ± 659. 20
RHCR + HM Cost 0% N/A N/A
RHCR + Traffic Flow 32% 3. 57 ± 0. 20 324. 12 ± 63. 91
RHCR + Unweighted 0% N/A N/A
```

```
9
```

```
DPP + CMA-ES 100% 5.17±0.00 28. 75 ± 0. 53
DPP + Crisscross 100% 4. 76 ± 0. 01 17.36±0.
DPP + HM Cost 100% 4. 32 ± 0. 01 29. 94 ± 1. 39
DPP + Traffic Flow 100% 4. 07 ± 0. 00 578. 43 ± 53. 61
DPP + Unweighted 100% 4. 34 ± 0. 01 20. 30 ± 0. 49
```

```
10
```

```
PIBT + CMA-ES 100% 4. 74 ± 0. 01 0. 58 ± 0. 00
PIBT + PIU 100% 4. 77 ± 0. 01 0. 58 ± 0. 01
PIBT + Unweighted 100% 3. 75 ± 0. 00 0.50±0.
RHCR + Unweighted 100% 4.95±0.00 87. 37 ± 0. 72
```

Table 3: Success rates (SR), throughput, and CPU runtimes of the
simulations on different guidance graphs. For RHCR and DPP, the
success rate is the percentage of simulations that end without con-
gestion. For PIBT, it is the percentage of simulations that end with-
out timeout. We measure the throughput and CPU runtime over only
successful simulations.

that they run the same number of simulations to ensure a fair
comparison. In particular, we set batch sizeb= 100and the
number of iterationsI = 100for both CMA-ES and PIU,
resulting in a total ofb×I= 10k objective function evalua-
tions for both algorithms. In each iteration, we select the top
Nbest= 50solutions to update the Gaussian distribution. For
CMA-ES, each evaluation runsNecma= 5simulations, re-

```
sulting in 50 k simulations. For PIU, each evaluation runs the
PIU algorithm forNp= 5iterations and each iteration runs
Nepiu= 1simulation, resulting in 50 k simulations, identical
to CMA-ES. We run each simulation for 1,000 timesteps. For
RHCR and DPP, we stop the simulation early in case of con-
gestion, which happens if more than half of the agents wait at
their current location.
Update Model. Given our use of grid maps in the experi-
ments, we use a Convolutional Neural Network (CNN) as our
update model, which can generate guidance graphs for maps
of arbitrary sizes. The CNN has 3 convolutional layers of
kernel sizes 3, 1, 1, respectively. Each layer is followed by a
ReLU activation and a batch normalization layer. The update
model has 4,231 parameters. For a map of dimensionh×w,
we represent the edge weightsωof the guidance graph as a
tensor of sizeh×w× 5 , where the first four channels are the
movement costs and the last channel is the wait costs.
Baselines. As mentioned in Section 3, we have 4 baseline
guidance graphs, namely (1)Unweighted, (2)Crisscross[Co-
hen, 2020], (3)HM Cost[Cohenet al., 2016], and (4)Traffic
Flow[Chenet al., 2024]. We discuss the methods of generat-
ing the baseline guidance graphs in Appendix A.
Evaluation. To evaluate PIU, we use the optimized update
model to run PIU withNepiu = 1to generate the guid-
ance graph. In Appendix B.2, we show that the choice of
Nepiudoes not have significant impact on the throughput of
the generated guidance graphs. When we evaluate a guidance
graph from CMA-ES, PIU, or baselines with a given number
of agents, we run 50 simulations, each for 1,000 timesteps,
and report the results with both means and standard errors.
Implementation. We implement the update model in Py-
Torch [Paszkeet al., 2019], CMA-ES in Pyribs [Tjanakaet
al., 2023], and Traffic Flow and HM Cost guidance graph
generation in Python. We implement the lifelong MAPF al-
gorithms in C++ based on openly available implementation
from previous works [Liet al., 2021; Okumuraet al., 2019].
Compute Resource. We run our experiments on two ma-
chines: (1) a local machine with a 64-core AMD Ryzen
Threadripper 3990X CPU, 192 GB of RAM, and an Nvidia
RTX 3090Ti GPU, and (2) a high-performing cluster with nu-
merous 64-core AMD EPYC 7742 CPUs, each with 256 GB
of RAM. We measure all CPU runtime on machine (1).
```

### 5.2 Results

```
GGO vs Baselines.We first compare our optimized guidance
graphs with the baseline guidance graphs. For each guidance
graph, we run 50 simulations and report the numerical re-
sults in Table 3 in the format ofx±y, wherexis the av-
erage andyis the standard error. Both CMA-ES and PIU
outperform all baseline guidance graphs in all setups in terms
of throughput. Specifically, CMA-ES outperforms all base-
line guidance graphs in all setups, showing the versatility of
the algorithm. For the baseline methods, the human-designed
crisscross guidance performs quite well in setups 2, 3, 4, 6,
7, 8, and 9, outperforming all other baselines. Traffic Flow
is more competitive in setups 1 and 5. When comparing the
throughput of CMA-ES and PIU, no clear winner emerges:
CMA-ES wins in setups 1, 2, 6, and 7, PIU wins in 3 and 4,
```

```
50 150 350
Number of Agents
```

```
0
```

```
4
```

```
8
```

```
Throughput
```

```
PIBT + CMA-ES
PIBT + PIU
PIBT + Unweighted
RHCR + Unweighted
```

Figure 5: Setup 10: An optimized guidance graph enables PIBT to
have competitive throughput with RHCR despite the advantage of
RHCR with fewer agents.

and they perform similarly in setup 5. Appendix C visualizes
all the optimized guidance graphs.
We also report CPU runtimes in Table 3, although run-
times are not the optimization objectives of our GGO algo-
rithms. PIBT runs very fast and finishes all 1,000-timestep
simulations in 5 seconds, implying each timestep takes less
than 0.005 seconds. Therefore, the runtime difference be-
tween different methods is negligible in practice. RHCR and
DPP are significantly slower than PIBT. For RHCR, CMA-
ES leads to the best runtime. For DPP, it runs PBS to solve
a one-shot MAPF instance for all agents in the first timestep
and then, in each timestep, replans only for agents that have
just reached their goals. We conjecture that the slower run-
time of DPP with CMA-ES than baselines comes from the
slower runtime in the first timestep. This is because the op-
timized guidance graph could encourage the agents to take
longer paths in order to avoid congestion.
To further understand the performance of the optimized
guidance graphs, we vary the number of agents and plot the
throughput in Figure 4. The trends are similar in all maps,
with CMA-ES and PIU generally outperforming all baselines,
except that Traffic Flow matches PIU with fewer agents In
random-32-32-20androom-64-64-8. However, Traffic Flow
is less competitive in all other maps, indicating that the per-
formance of Traffic Flow depends on the map structures.
RHCR vs PIBT+GGO.Given the considerable runtime ad-
vantage of PIBT over RHCR (and DPP), we conduct an
additional experiment, as detailed in setup 10 of Table 2,
to explore whether PIBT with an optimized guidance graph
can achieve higher throughput than RHCR without guidance
graphs. We chooseNa= 150because it is the largest num-
ber with which RHCR without guidance graphs can maintain
a 100% success rate. Setup 10 in Table 3 shows the numer-
ical results. While RHCR still has the highest throughput,
both GGO methods significantly reduce the throughput gap
between PIBT and RHCR, from 24.2% to less than 4.2%.
Therefore, with the help of our optimized guidance graph,
we can enable a greedy, distributed, yet extremely fast rule-
based MAPF algorithm (PIBT) to achieve performance com-
parable to a centralized, computationally heavy, search-based
MAPF algorithm (RHCR). Figure 5 further compares PIBT
and RHCR with various numbers of agents. The throughput
of RHCR quickly drops after 150 agents, while that of PIBT
maintains an increasing trend with more agents.
Update Model Transferability.We attempt to transfer the
update model optimized with setup 2 to larger warehouse

```
45 × 47 57 × 58 69 × 69 81 × 80 93 × 91
Map Size
```

```
6
```

```
7
```

```
8
```

```
9
```

```
10
```

```
11
```

```
Max Throughput
```

```
45 × 47 57 × 58 69 × 69 81 × 80 93 × 91
Map Size
```

```
1000
```

```
2000
```

```
3000
```

```
# Agents at Max Throughput
```

```
Figure 6: Max throughput and the number of agents at this maxi-
mum. PIU Transfer refers to using the update model optimized in
setup 2 to generate guidance graphs.
```

```
maps with similar layouts. We mimic the layout pattern of
warehouse-33-36and design larger maps with sizes up to 93
×91 by repeating blocks of 10 shelves and endpoints and
placing workstations on the left and right borders. Figure 9
in Appendix B.3 plots the resulting maps. We use the opti-
mized update model from setup 2 to generate guidance graphs
for these maps with an increasing number of agents, ranging
from around 10% to 90% of the non-black tiles in the maps.
We then run 50 simulations in each of the generated guid-
ance graphs with PIBT. We plot the maximum throughput
achieved in each map and the corresponding number of agents
in Figure 6, comparing PIU Transfer with baseline guidance
graphs. We observe that PIU Transfer dominates all baselines
with all sizes in terms of throughput. Notably, while criss-
cross has the second highest throughput across different map
sizes, PIU Transfer can achieve higher throughput with an
equal or smaller number of agents. Appendix B.3 shows the
throughput with different numbers of agents in larger ware-
house maps.
```

## 6 Conclusion

```
We define guidance graphs and GGO to maximize the
throughput of lifelong MAPF, reviewing previous works on
guidance in MAPF and highlighting the generality of our
guidance graph. We propose CMA-ES and PIU that opti-
mize guidance graphs across different algorithms and maps.
We also show that the update model can generate guidance
graphs for larger maps with similar patterns.
Our work is limited in many ways, yielding numerous
future directions. First, both CMA-ES and PIU are com-
putationally expensive, requiring a large number of lifelong
MAPF simulations, taking 1.2 hours (setup 1) to 55 hours
(setup 8) to run on machine (2) in Section 5.1. Future works
can focus on reducing the computational requirements of both
methods. Second, our optimized guidance graphs improve
throughput but such improvement lacks explainability. Fu-
ture works can focus on either generating more explainable
guidance graphs or analyzing the explainability of our opti-
mized guidance graphs. Third, although our guidance graph
can be used with online update mechanisms introduced in
previous works [Chenet al., 2024; Yu and Wolf, 2023], we
limit our experiment settings without such mechanisms. In-
tegrating these mechanisms with our GGO algorithms could
further enhance MAPF guidance utility.
```

## Acknowledgments

This work used Bridge- 2 at Pittsburgh Supercomputing Cen-
ter (PSC) through allocation CIS 220115 from the Advanced
Cyberinfrastructure Coordination Ecosystem: Services &
Support (ACCESS) program, which is supported by National
Science Foundation grants # 2138259 , # 2138286 , # 2138307 ,

# 2137603 , and # 2138296.

## References

Max Barer, Guni Sharon, Roni Stern, and Ariel Felner. Sub-
optimal variants of the conflict-based search algorithm for
the multi-agent pathfinding problem. InProceedings of
the Annual Symposium on Combinatorial Search (SoCS),
pages 19–27, 2014.

Rafał Biedrzycki. Handling bound constraints in CMA-ES:
An experimental study.Swarm and Evolutionary Compu-
tation, 52:100627, 2020.

Michal Cap, Jir ́ ́ı Vokr ́ınek, and Alexander Kleiner. Com-
plete decentralized method for on-line multi-robot trajec-
tory planning in well-formed infrastructures. InProceed-
ings of the International Conference on Automated Plan-
ning and Scheduling (ICAPS), pages 324–332, 2015.

Zhe Chen, Daniel Harabor, Jiaoyang Li, and Peter Stuckey.
Traffic flow optimisation for lifelong multi-agent path find-
ing. InProceedings of the AAAI Conference on Artificial
Intelligence (AAAI), pages 20674–20682, 2024.

Liron Cohen, Tansel Uras, and Sven Koenig. Feasibility
study: Using highways for bounded-suboptimal multi-
agent path finding. InProceedings of the International
Symposium on Combinatorial Search (SoCS), pages 2–8, 2015.

Liron Cohen, Tansel Uras, T. K. Satish Kumar, Hong Xu,
Nora Ayanian, and Sven Koenig. Improved solvers for
bounded-suboptimal multi-agent path finding. InProceed-
ings of the International Joint Conference on Artificial In-
telligence (IJCAI), pages 3067–3074, 2016.

Liron Cohen. Efficient Bounded-Suboptimal Multi-Agent
Path Finding and Motion Planning via Improvements to
Focal Search. PhD thesis, University of Southern Califor-
nia, 2020.

Mehul Damani, Zhiyao Luo, Emerson Wenzel, and Guil-
laume Sartoretti. PRIMAL 2 : Pathfinding via reinforce-
ment and imitation multi-agent learning - lifelong. IEEE
Robotics and Automation Letters, 6(2):2666–2673, 2021.

Florian Grenouilleau, Willem-Jan van Hoeve, and John N.
Hooker. A multi-label A\* algorithm for multi-agent
pathfinding. InProceedings of the International Con-
ference on Automated Planning and Scheduling (ICAPS),
pages 181–185, 2019.

Shuai D. Han and Jingjin Yu. Optimizing space utilization for
more effective multi-robot path planning. InProceedings
of the International Conference on Robotics and Automa-
tion (ICRA), pages 10709–10715, 2022.

Nikolaus Hansen, yoshihikoueno, ARF1, Gabriela Kadle-
cova, Kento Nozawa, Luca Rolshoven, Matthew Chan, ́

```
Youhei Akimoto, brieglhostis, and Dimo Brockhoff.
CMA-ES/pycma: r3.3.0, January 2023.
Nikolaus Hansen. The CMA evolution strategy: A tutorial.
ArXiv, abs/1604.00772, 2016.
M. Renee Jansen and Nathan R Sturtevant. Direction maps
for cooperative pathfinding. InProceedings of the AAAI
Conference on Artificial Intelligence and Interactive Digi-
tal Entertainment (AIIDE), pages 185–190, 2008.
Daphne Koller and Nir Friedman. Probabilistic Graphical
Models: Principles and Techniques - Adaptive Computa-
tion and Machine Learning. The MIT Press, 2009.
Ngai Meng Kou, Cheng Peng, Hang Ma, T. K. Satish Kumar,
and Sven Koenig. Idle time optimization for target assign-
ment and path finding in sortation centers. InProceedings
of the AAAI Conference on Artificial Intelligence (AAAI),
pages 9925–9932, 2020.
Ming-Feng Li and Min Sun. The study of highway for life-
long multi-agent path finding.ArXiv, 2304.04217, 2023.
Jiaoyang Li, Kexuan Sun, Hang Ma, Ariel Felner, T. K. Satish
Kumar, and Sven Koenig. Moving agents in formation in
congested environments. InProceedings of the Interna-
tional Joint Conference on Autonomous Agents and Multi-
agent Systems (AAMAS), pages 726–734, 2020.
Jiaoyang Li, Andrew Tinka, Scott Kiesel, Joseph W. Durham,
T. K. Satish Kumar, and Sven Koenig. Lifelong multi-agent
path finding in large-scale warehouses. InProceedings
of the AAAI Conference on Artificial Intelligence (AAAI),
pages 11272–11281, 2021.
Minghua Liu, Hang Ma, Jiaoyang Li, and Sven Koenig.
Task and path planning for multi-agent pickup and deliv-
ery. InProceedings of the International Conference on
Autonomous Agents and Multi-Agent Systems (AAMAS),
pages 1152–1160, 2019.
Hang Ma, Jiaoyang Li, T. K. Satish Kumar, and Sven Koenig.
Lifelong multi-agent path finding for online pickup and de-
livery tasks. InProceedings of the International Confer-
ence on Autonomous Agents and Multiagent Systems (AA-
MAS), pages 837–845, 2017.
Hang Ma, Jingxing Yang, Liron Cohen, T. K. Satish Ku-
mar, and Sven Koenig. Feasibility study: Moving non-
homogeneous teams in congested video game environ-
ments. InProceedings of the AAAI Conference on Ar-
tificial Intelligence and Interactive Digital Entertainment
(AIIDE), pages 270–272, 2017.
Hang Ma, Daniel Harabor, Peter J. Stuckey, Jiaoyang Li,
and Sven Koenig. Searching with consistent prioritization
for multi-agent path finding. InProceedings of the AAAI
Conference on Artificial Intelligence (AAAI), pages 7643–
7650, 2019.
Keisuke Okumura, Manao Machida, Xavier D ́efago, and Ya-
sumasa Tamura. Priority inheritance with backtracking for
iterative multi-agent path finding. InProceedings of the In-
ternational Joint Conference on Artificial Intelligence (IJ-
CAI), pages 535–542, 2019.
```

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer,
James Bradbury, Gregory Chanan, Trevor Killeen, Zeming
Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison,
Andreas Kopf, Edward Yang, Zachary DeVito, Martin Rai-
son, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner,
Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: An
imperative style, high-performance deep learning library.
InProceedings of the Advances in Neural Information Pro-
cessing Systems (NeurIPS), pages 8024–8035, 2019.

Mike Phillips and Maxim Likhachev. SIPP: Safe interval path
planning for dynamic environments. InProceedings of the
IEEE International Conference on Robotics and Automa-
tion (ICRA), pages 5628–5635, 2011.

Roni Stern, Nathan R. Sturtevant, Ariel Felner, Sven Koenig,
Hang Ma, Thayne T. Walker, Jiaoyang Li, Dor Atzmon,
Liron Cohen, T. K. Satish Kumar, Roman Bartak, and Eli ́
Boyarski. Multi-agent pathfinding: Definitions, variants,
and benchmarks. InProceedings of the International Sym-
posium on Combinatorial Search (SoCS), pages 151–159, 2019.

Bryon Tjanaka, Matthew C. Fontaine, David H. Lee, Yu-
lun Zhang, Nivedit Reddy Balam, Nathaniel Dennler, Su-
jay S. Garlanka, Nikitas Dimitri Klapsis, and Stefanos
Nikolaidis. pyribs: A bare-bones python library for quality
diversity optimization. InProceedings of the Genetic and
Evolutionary Computation Conference (GECCO), pages
220–229, 2023.

Sumanth Varambally, Jiaoyang Li, and Sven Koenig. Which
MAPF model works best for automated warehousing? In
Proceedings of the Symposium on Combinatorial Search
(SoCS), pages 190–198, 2022.

Konstantinos Varelas, Anne Auger, Dimo Brockhoff, Niko-
laus Hansen, Ouassim Ait ElHara, Yann Semet, Rami
Kassab, and Fred ́eric Barbaresco. A comparative study of ́
large-scale variants of CMA-ES. InProceedings of the In-
ternational Conference on Parallel Problem Solving from
Nature (PPSN), pages 3–15, 2018.

Qian Wan, Chonglin Gu, Sankui Sun, Mengxia Chen, Hejiao
Huang, and Xiaohua Jia. Lifelong multi-agent path finding
in a dynamic environment. InProceedings of the Interna-
tional Conference on Control, Automation, Robotics and
Vision (ICARCV), pages 875–882, 2018.

Ko-Hsin Cindy Wang and Adi Botea. Fast and memory-
efficient multi-agent pathfinding. InProceedings of the
International Conference on Automated Planning and
Scheduling (ICAPS), pages 380–387, 2008.

Jingjin Yu and Steven M. LaValle. Structure and intractabil-
ity of optimal multi-robot path planning on graphs. InPro-
ceedings of the AAAI Conference on Artificial Intelligence
(AAAI), pages 1444–1449, 2013.

Ge Yu and Michael Wolf. Congestion prediction for large
fleets of mobile robots. InProceedings of the Interna-
tional Conference on Robotics and Automation (ICRA),
pages 7642–7649, 2023.

Yulun Zhang, Matthew C. Fontaine, Varun Bhatt, Stefanos
Nikolaidis, and Jiaoyang Li. Arbitrarily scalable environ-

```
ment generators via neural cellular automata. InProceed-
ings of the Advances in Neural Information Processing
Systems (NeurIPS), pages 57212–57225, 2023.
Yulun Zhang, Matthew C. Fontaine, Varun Bhatt, Stefanos
Nikolaidis, and Jiaoyang Li. Multi-robot coordination and
layout design for automated warehousing. InProceedings
of the International Joint Conference on Artificial Intelli-
gence (IJCAI), pages 5503–5511, 2023.
```

```
Algorithm 2:Traffic Flow Guidance Graph Genera-
tion
Input: Gg(Vg,Eg,ω), whereω= 1
Nbase: number of iterations
S,T⊂Vg: potential start and goal locations of
the agents
planpath: function to find single agent path
given start, goal, and guidance graph.
1 Initialize vertex usageUVg(v)← 0 ,∀v∈Vg
2 Initialize edge usageUEg(u,v)← 0 ,∀(u,v)∈Eg
3 fori← 1 toNbasedo
4 Samplesi∈RS,gi∈RTs.t.si̸=gi
5 Pi←planpath(si,gi,Gg)
6 forVertexv∈Pido
7 UVg(v)←UVg(v) + 1
8 forEdge(u,v)∈Pido
9 UEg(u,v)←UEg(u,v) + 1
```

10 forEdge(u,v)∈Egdo

```
11 p(v)←⌈
```

```
UVg(v)− 1
2 ⌉
12 c(u,v)←UEg(u,v)×UEg(v,u)
13 ω(u,v)←1 +c(u,v) +p(v)
```

14 ωT F←ω
15 ReturnGT F(Vg,Eg,ωT F)

## A Baseline Guidance Graphs

```
In this section, we discuss the baseline guidance graphs that
we compare with in Section 5. As mentioned in Section 3, we
have 4 baseline guidance graphs, namely (1)Unweighted, (2)
Crisscross[Cohen, 2020], (3)HM Cost[Cohenet al., 2016],
and (4)Traffic Flow[Chenet al., 2024].
```

### A.1 Unweighted and Crisscross

```
Both unweighted and crisscross guidance graphs are human-
designed. We define unweighted guidance graph as follows:
```

```
Definition 6(Unweighted Guidance Graph). We define a
guidance graphG(Vg,Eg,ω)whereω= 1as an unweighted
guidance graph.
Our crisscross guidance graph follows the definition of
crisscross highways [Cohen, 2020]. In particular:
```

```
Definition 7(Crisscross Guidance Graph). Given an un-
weighted guidance graphG(Vg,Eg,ω) for a 4-neighbor
grid-based map in which agents can move up, down, left, or
right at each vertex, we select a subset of edgesEc⊂Eg
such that:
```

1. in the even rows, all edges pointing right are chosen,
2. in the odd rows, all edges pointing left are chosen,
3. in the even columns, all edges pointing up are chosen,
4. in the odd columns, all edges pointing down are chosen.

```
We let the edge weights of edges inEcbe 0.5 and all other
edgesEg\Ecbe 1, promoting the agents to use edges inEc.
```

```
Algorithm 3:HM Cost Guidance Graph Generation
Input: Gg(Vg,Eg,ω), whereω= 1
Nbase: number of iterations
S,T⊂Vg: potential start and goal locations of
the agents
planpath: function to find single agent path
given start, goal, and guidance graph.
α,β,γ∈R: hyperparameters of computing
follow preferencep, interference costt, and
saturation costt, respectively.
1 Initialize vertex usageUVg(v)← 0 ,∀v∈Vg
2 Initialize edge usageUEg(u,v)← 0 ,∀(u,v)∈Eg
3 fori← 1 toNbasedo
4 Samplesi∈RS,gi∈RTs.t.si̸=gi
5 Pi←planpath(si,gi,Gg)
6 forVertexv∈Pido
7 UVg(v)←UVg(v) + 1
8 forEdge(u,v)∈Pido
9 UEg(u,v)←UEg(u,v) + 1
10 forEdge(u,v)∈Egdo
11 p(u,v)←α×
```

```
UEg(u,v)
Nbase
12 t(u,v)←β×
```

```
UEg(v,u)
Nbase
```

```
13 s(u,v)←γ
```

```
UEg(u,v)+UEg(v,u)
2 Nbase
14 c(u,v)← 1 −p(u,v) +t(u,v) +s(u,v)
15 ω(u,v)←c(u,v)
```

```
16 SetEHM⊂Egwith lowestc(u,v)s.t.
|EHM|=^17 |Eg|andu̸=v
17 SetE′HM←randomly sample^15 of edges fromEHM
18 Initialize edge weightsωHM
19 forEdge(u,v)∈Egdo
20 if(u,v)∈EHM′ then
21 ωHM(u,v)← 0. 5
22 else
23 ωHM(u,v)← 1
```

```
24 ReturnGHM(Vg,Eg,ωHM)
```

### A.2 Traffic Flow and HM Cost

```
The previous work on traffic flow guidance [Chenet al.,
2024] is developed for lifelong MAPF with an online up-
date mechanism. The work on HM Cost guidance [Cohen
et al., 2016] is developed for one-shot MAPF. Therefore, we
adapt both methods to construct guidance graphs for lifelong
MAPF.
Traffic Flow.Algorithm 2 describes the adapted Traffic Flow
guidance graph generation procedure. On a high level, we
iteratively plan single-agent paths based on the current guid-
ance graph and update the graph based on the congestion of
these paths. Starting with an unweighted guidance graph, we
sample a pair of start and goal locations (Line 4) and search
for a pathPithat minimizes the sum of its edge weights on
the current guidance graph (Line 5). Then, we increment
the usages of vertices and edges onPiby 1 (Lines 6 to 9).
```

```
200 400 800
Number of Agents
```

```
0
```

```
4.
```

```
9
```

```
Throughput
```

```
PIBT + CMA-ES (Normalization)
PIBT + CMA-ES (Projection)
PIBT + CMA-ES (Reflection)
PIBT + CMA-ES (Transformation)
```

```
(a) Setup 1:random-32-32-
```

```
50 150 350
Number of Agents
```

```
0
```

```
4
```

```
8
```

```
Throughput
```

```
RHCR + CMA-ES (Normalization)
RHCR + CMA-ES (Projection)
RHCR + CMA-ES (Reflection)
RHCR + CMA-ES (Transformation)
```

```
(b) Setup 8:warehouse-33-
```

Figure 7: Comparison of different bounds handling methods in
CMA-ES in setup 1 and 4. The black vertical lines indicates the
number of agentsNaused to optimize the guidance graph.

Afterward, we follow the previous work [Chenet al., 2024]
to compute the vertex congestion averaged over vertex us-
age for each vertex (Line 11) and the contraflow congestion
for each edge (Line 12) and then update the edge weights of
the guidance graph by summing the vertex congestion, the
contraflow congestion, and 1 (Line 13), where the 1 indi-
cates the zero congestion cost. The updated edge weights
inflate the cost of frequently used edges so that the following
single-agent paths are encouraged to avoid these edges. Traf-
fic Flow repeats the procedure forNbaseiterations and returns
the edge weights from the last iteration as the guidance graph
(Lines 14 and 15). Since the Traffic Flow algorithm does not
consider wait costs, we set wait costs of all vertices to be 1.

HM Cost.Algorithm 3 describes the adapted HM Cost guid-
ance graph generation procedure. Similar to Traffic Flow, we
first sample start and goal locations (Line 4) and run single-
agent path planning on the current guidance graph (Line 5),
updating vertex and edge usage (Lines 6 to 9). Then we fol-
low the previous work [Cohenet al., 2016] to compute (1) the
follow preferencep(Line 11), which encourages the agents
to traverse through previously used edges, (2) the interference
costt, which discourages the agents to traverse through edges
in the opposite directions of previously used edges, and (3)
the saturation costs. The HM costcis a linear combination
of the above three variables (Line 14), with smaller HM costs
indicating edges that are used more frequently. We then set
the HM cost as the edge weights of the guidance graph and
repeat the iteration forNbasetimes (Line 15).

Nevertheless, after runningNbasesingle agent path plan-
ning, HM Cost includes additional procedures to generate the
final guidance graph from the computed HM cost. Following
previous work [Cohenet al., 2016], we select the top^17 of the
edges with the smallest HM cost and then randomly sample
1
5 of them as the set of highway edges (Lines 16 and 17).
We then set the weights of the highway edges as 0.5 and
non-highway edges as 1, forming a guidance graph (Lines 18
to 23). This is equivalent to settingc= 2following the def-
inition of highway in Section 3 and previous works [Cohen
et al., 2015, 2016; Li and Sun, 2023] (i.e. the weights of the
highway edges are 1 and non-highway edges 2). In HM Cost,
all selected edges are not self-edges. Therefore, the wait costs
of all vertices are 1, same as the non-highway edges.

```
50 400 800
Number of Agents
```

```
0
```

```
4
```

```
8
```

```
Throughput
```

```
PIBT + PIU (Ne piu= 1)
PIBT + PIU (Ne piu= 10)
PIBT + PIU (Ne piu= 30)
PIBT + PIU (Ne piu= 50)
```

```
Figure 8: Effect of different values ofNepiu. The black vertical line
indicates the number of agents used to optimize the update model.
```

### A.3 Hyperparameters for Baseline Methods

```
In both Traffic Flow and HM Cost, we useNbase= 10, 000
to generate the guidance graphs for all maps. In HM Cost, we
follow the previous work [Cohenet al., 2016] to useα= 0. 5 ,
β= 1. 2 , andγ= 1. 3.
```

## B Additional Experiments

```
We include the following additional experiments: (1) for
bounds handling of CMA-ES, we show ablation experiments
on normalization comparing with other representative bounds
handling methods presented in [Biedrzycki, 2020], (2) for
PIU, we test guidance graph generation with different val-
ues ofNepiuto test if running more simulations in PIU can
improve the generated guidance graph, and (3) the through-
put with different number of agents in large warehouse maps
shown in Figure 9.
```

### B.1 Bounds Handling in CMA-ES

```
A previous work [Biedrzycki, 2020] compares a number of
bounds handling approaches for CMA-ES. Their compari-
son demonstrates that resampling, reflection, projection, and
transformation are among the most popular and empirically
best choices of bounds handling methods. Therefore, we first
briefly present these methods and compare them with our pro-
posed bounds handling, namely min-max normalization. For
simplicity, we use normalization to refer to min-max normal-
ization in the following texts.
```

```
Bounds-Handling Methods
Assume that we optimize forω′∈Rnsuch thatl≤ω′≤u
withl,u∈Rn. Then, given a randomly sampled solution
ω ∈Rn, the bounds handling methods seek to generate a
valid solutionω′fromω.
Resampling: The resampling method keeps resampling from
the Gaussian distribution until all variablesωi∈ωare within
the given bounds.
Notably, the resampling method does not stop until all vari-
ables are within the bounds. The probability of sampling a
solution from a Gaussian distribution such that all variables
are within a given bound depends heavily on the parameters
of the distribution and the dimensionality of the search space.
Our experiment setups specified in Table 2 have at least 1159
parameters, making resampling inapplicable.
```

```
(a) 45× 47 (b) 57× 58 (c) 69× 69 (d) 81× 80 (e) 93× 91
```

```
Figure 9: Warehouse maps used for testing the transferability of the update model optimized with setup 2 in Section 5.
```

```
PIBT + Crisscross PIBT + HM Cost PIBT + PIU Transfer PIBT + Traffic Flow PIBT + Unweighted
```

(^0200) Number of Agents 850 1500
5
10
Throughput
(a) 45× 47
(^0200) Number of Agents 1200 2200
5
10
Throughput
(b) 57× 58
(^0400) Number of Agents 1700 3000
5
10
Throughput
(c) 69× 69
(^0600) Number of Agents 2300 4000
6
12
Throughput
(d) 81× 80
(^01000) Number of Agents 3500 6000
6
12
Throughput
(e) 93× 91
Figure 10: Throughput with different numbers of agents in larger warehouse maps.
Projection: The projection method projects out-of-bounds
solutions to the lower or upper bounds.
ω′i=

#### 

#### 

#### 

```
ωi li≤ωi≤ui
li ωi<li
ui ωi>ui
```

#### (4)

Reflection: The reflection method reflects the out-of-bounds
solutions to within the bounds such that:

```
ω′i=
```

#### 

#### 

#### 

```
ωi li≤ωi≤ui
2 li−ωi ωi<li
2 ui−ωi ωi>ui
```

#### (5)

Transformation: The transformation method maps out-of-
bounds solutions to within the bounds such that:

```
ali=min(
```

```
ui−li
2
```

#### ,

```
1 +|li|
20
```

#### ) (6)

```
aui=min(
```

```
ui−li
2
```

#### ,

```
1 +|ui|
20
```

#### ) (7)

```
ω′i=
```

#### 

#### 

#### 

#### 

#### 

```
ωi li+ali≤ωi≤ui−aui
li+(ωi−(li−a
```

```
li)) 2
4 ali li−a
```

```
l
i≤ωi<li+a
```

```
l
i
ui−
```

```
(ωi−(ui+aui))^2
4 aui ui−a
```

```
u
i<ωi≤ui+a
```

u
i
(8)
Intuitively, the transformation does not change the sampled
solution if it is within the bound[li+ali,ui−aui]. If the solu-
tion falls into[li−ali,li+ali)or(ui−aui,ui+aui], quadratic
transformations are applied to map the solution to within the

```
bounds. If the solution is smaller thanli−alior larger than
ui+aui, the transformation method first uses Equation (5) to
reflect the solution usingli−aliorui+aui as the bounds.
Then Equation (8) is applied to further transform the value if
necessary. We use the transformation method implemented in
Pycma [Hansenet al., 2023] to run the experiments.
```

```
Empirical Comparison
```

```
We compare normalization with projection, reflection, and
transformation on setup 1 and setup 4. Similar to Section 5,
we run lifelong MAPF simulations with varying numbers of
agents, each with 50 simulations. Figure 7 shows the through-
put. For both setups, normalization empirically achieves the
highest throughput withNaagents. While scaling to more
agents, normalization consistently has better throughput than
other bounds handling methods in setup 1 with PIBT. While
the throughput of RHCR drops more rapidly with normaliza-
tion afterNaagents, this can be compensated by increasing
Naduring the optimization of the guidance graph.
```

```
The advantage of normalization comes from the utilization
of the guidance graph in lifelong MAPF. As discussed in Sec-
tion 4, the absolute magnitude of the edge weights has less
or no impact on the MAPF solutions compared to the rela-
tive magnitude. Therefore, normalizing these edge weights
simplifies the optimization problem, shifting the focus to
theshapeof the Gaussian distribution modeling the edge
weights, rather than their precise numerical values. This sim-
plification enables normalization to outperform other bounds-
handling methods for CMA-ES.
```

```
(a) Setup 1 (random-32-32-20): PIBT + CMA-ES
```

```
(b) Setup 1 (random-32-32-20): PIBT + PIU
```

```
(c) Setup 2 (warehouse-33-36): PIBT + CMA-ES
```

```
(d) Setup 2 (warehouse-33-36): PIBT + PIU
```

```
Figure 11: Optimized guidance graphs of setups 1 and 2.
```

### B.2 On the Value ofNepiu

We use the update model optimized with setup 2 to gener-
ate guidance graphs withNepiu∈ { 1 , 10 , 30 , 50 }. We then
evaluate the guidance graphs by running lifelong MAPF sim-
ulations with various number of agents, each with 50 simula-
tions. We show the result in Figure 8. We find no significant
difference in throughput with different values ofNepiu.

### B.3 Throughput of Larger Warehouse Maps

Figure 9 shows the maps used for the transferability experi-
ment. Figure 10 shows throughput with different number of
agents in large warehouse maps shown in Figure 9. In gen-
eral, we observe that PIU Transfer dominates all baselines.

## C Optimized Guidance Graphs

We show the optimized guidance graphs with CMA-ES and
PIU in Figures 11 to 14. In general, the guidance graphs opti-
mized by CMA-ES are hardly explainable. Those optimized

```
by PIU possess more patterns. For example, inempty-48-
(Figure 12d), the optimized wait costs are lower than move-
ment costs, promoting agents to wait instead of moving in
case of congestion. Inrandom-64-64-20(Figure 13b), on the
other hand, the wait costs are generally larger than movement
costs. Nevertheless, it is non-trivial to explain why such guid-
ance graphs improve throughput in these maps. We leave gen-
erating explainable guidance graphs or explaining optimized
guidance graphs to future works.
```

```
(a) Setup 3 (maze-32-32-4): PIBT + CMA-ES
```

```
(b) Setup 3 (maze-32-32-4): PIBT + PIU
```

```
(c) Setup 4 (empty-48-48): PIBT + CMA-ES
```

```
(d) Setup 4 (empty-48-48): PIBT + PIU
```

```
(e) Setup 5 (room-64-64-8): PIBT + CMA-ES
```

```
(f) Setup 5 (room-64-64-8): PIBT + PIU
```

Figure 12: Optimized guidance graphs of setups 3, 4, and 5.

```
(a) Setup 6 (random-64-64-20): PIBT + CMA-ES
```

```
(b) Setup 6 (random-64-64-20): PIBT + PIU
```

```
(c) Setup 7 (den312d): PIBT + CMA-ES
```

```
(d) Setup 7 (den312d): PIBT + PIU
```

```
(e) Setup 8 (warehouse-33-36): RHCR + CMA-ES
```

Figure 13: Optimized guidance graphs of setups 6, 7, and 8.

```
(a) Setup 9 (warehouse-20-17): DPP + CMA-ES
```

(b) Setup 10 (warehouse-33-36): PIBT + CMA-ES (150 agents)

```
(c) Setup 10 (warehouse-33-36): PIBT + PIU (150 agents)
```

```
Figure 14: Optimized guidance graphs of setups 9 and 10.
```
