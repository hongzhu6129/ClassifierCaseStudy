package RedWineQuality;

public class RedWineFeatures {
	
	public double fixedAcidity; 		/* min=4.6, 	max=15.9, range=[4,16], stepSize=0.6 	*/
	public double volatileAcidity; 		/* min=0.12, 	max=1.58, range=[0,2],	stepSize=0.1 	*/
	public double citricAcid;			/* min=0, 		max=1,	  range=[0,2],	stepSize=0.1 	*/
	public double residualSugar;		/* min=0.9, 	max=15.1, range=[0,16], stepSize=0.8	*/
	public double chlorides;			/* min=0.01, 	max=0.61, range=[0,1],  stepSize=0.05	*/
	public double freeSulfurDioxide;	/* min=1, 		max=72,	  range=[0,100],stepSize=5		*/
	public double totalSulfurDioxide;	/* min=6,		max=289,  range=[0,300],stepSize=15		*/
	public double density;				/* min=0.99,	max=1,	  range=[0,2],	stepSize=0.1	*/
	public double pH;					/* min=2.74,	max=4.01, range=[2,5], 	stepSize=0.15	*/
	public double sulphates;			/* min=0.33, 	max=2,	  range=[0,3],	stepSize=0.15	*/
	public double alcohol;				/* min=8.4, 	max=14.9, range=[8,18], stepSize=0.5	*/			

	public RedWineFeatures() {
		fixedAcidity = 4.5; 
		volatileAcidity = 0.1; 
		citricAcid = 0.0;
		residualSugar = 0.5;
		chlorides = 0.0;
		freeSulfurDioxide = 1.0;
		totalSulfurDioxide = 5.0;
		density = 0.5;
		pH = 2.0;
		sulphates = 2.5; 
		alcohol = 8.0;
	}
	
	public RedWineFeatures(
			double p_fixedAcidity,
			double p_volatileAcidity,
			double p_citricAcid,
			double p_residualSugar,
			double p_chlorides,
			double p_freeSulfurDioxide,
			double p_totalSulfurDioxide,
			double p_density,
			double p_pH,
			double p_sulphates,
			double p_alcohol) {
		fixedAcidity = p_fixedAcidity;
		volatileAcidity = p_volatileAcidity;
		citricAcid = p_citricAcid;
		residualSugar = p_residualSugar;
		chlorides = p_chlorides;
		freeSulfurDioxide = p_freeSulfurDioxide;
		totalSulfurDioxide = p_totalSulfurDioxide;
		density = p_density; 
		pH = p_pH;
		sulphates = p_sulphates;
		alcohol = p_alcohol;
	}
	
	public String toString() {
		String str = "<" 
				+ fixedAcidity + "|"
				+ volatileAcidity + "|"
				+ citricAcid + "|"
				+ residualSugar + "|"
				+ chlorides + "|"
				+ freeSulfurDioxide + "|"
				+ totalSulfurDioxide + "|"
				+ density + "|"
				+ pH + "|"
				+ sulphates +"|"
				+ alcohol + ">";
		return str;
	} 
	
	public void valueOf(String str) {
		str = str.substring(1, str.length()-1);
		String vStr[] = str.split("\\|");
		fixedAcidity =	Double.parseDouble(vStr[0]);
		volatileAcidity = Double.parseDouble(vStr[1]);
		citricAcid = Double.parseDouble(vStr[2]);
		residualSugar = Double.parseDouble(vStr[3]);
		chlorides = Double.parseDouble(vStr[4]);	
		freeSulfurDioxide = Double.parseDouble(vStr[5]);
		totalSulfurDioxide = Double.parseDouble(vStr[6]);
		density = Double.parseDouble(vStr[7]);
		pH = Double.parseDouble(vStr[8]);
		sulphates = Double.parseDouble(vStr[9]);
		alcohol = Double.parseDouble(vStr[10]);
	}
}
