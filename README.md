# DHL.1
This is Part 1 of DHL Initial Analysis
Part 1:
Introduction:

Recommend the best MFG and DC along with six candidates in Newark, Atlanta, Dallas, Seattle, LA, and Chicago to get min fixed cost, min var cost, inbound and outbound cost. 2M bottles volume needs to be distributed into 30 major markets in the United States. 

Methodology

The following OR model used PuLP and the MIP model to solve for optimality with a 20% profit margin

Assumptions and Variables

Demand: Fixed annual demand of 2M bottles, allocated to 30 cities. We assume everything is logical and no shortages or backlogs. Volumes can be split across DCs for flexibility

Facility Roles: Each candidate site can host MFG, DC, both, or neither.

Inbound Cost: Full truckloads of 2000 bottles at $3 per mile. Approximate the # of truckloads as volume / 2,000

Outbound Cost: Shipments of 6 bottles per order, with one shipment per order. Fixed cost $9.75 per shipment + $3.50 per 500 miles. Distribution processing cost per order varies by DC

Total Cost: Shipments of 6 bottles per order, with one shipment per order. Fixed cost $9.75 per shipment + $3.50 per 500 miles. Distribution processing cost per order varies by DC. Distances use provided matrices; assumes direct parcel shipping, no intermediate hubs



OR Model
I formulated a capacitated facility location model with transportation costs, a standard OR approach for network design, with the following variables using Dantzig-Wolfe Decomposition



Results
The optimized network yields a total annual cost of $12,196,819.33. Key recommendations:
	Open Facilities: 
	Manufacturing: Dallas, TX (only). Fixed cost: $700,000; variable cost: $1.80/bottle (competitive among candidates).
	Distribution: Dallas, TX (only; co-located with MFG to eliminate inbound costs).
	Flows: 
	All 2,000,000 bottles produced and distributed from Dallas (self-supply: 2,000,000 bottles inbound to its DC).
	Dallas serves all 30 customer markets (no splits needed; central location minimizes average outbound distance ~1,000 miles).

Cost Component	Annual Cost ($)	% of Total
Fixed (MFG + DC)	1,050,000	8.6%
Variable Manufacturing	3,600,000	29.5%
Inbound Transportation	0	0% (co-location)
Outbound (Shipping + Processing)	7,546,819	61.9%
Total	12,196,819	100%

