package BankChurners;

public class BankChurnersValue {
	public int[] discValue;
	public int[] intValue;
	public double[] realValue;
	
	public BankChurnersValue() {
		discValue = new int[5];
		for (int i=0; i<=4; i++) {
			discValue[i]=0;
		}
		intValue = new int[11];
		for (int i=0; i<10; i++) {
			intValue[i] =0;
		}
		realValue = new double[3];
		realValue[0] = 0.0;
		realValue[1] = 0.0;
		realValue[2] = 0.0;
	}
	
	public BankChurnersValue(int[] dv, int[] iv, double[] rv) {
		discValue = new int[5];
		for (int i=0; i<=4; i++) {
			discValue[i]= dv[i];
		}
		intValue = new int[11];
		for (int i=0; i<10; i++) {
			intValue[i] = iv[i];
		}
		realValue = new double[3];
		realValue[0] = rv[0];
		realValue[1] = rv[1];
		realValue[2] = rv[2];
	}
	
	public String toString() {
		String str = "";
		for (int i=0; i<=4; i++) {
			str = str +"|" + discValue[i];
		}
		for (int i=0; i<=10; i++) {
			str = str +"|" + intValue[i];
		}
		str = str + "|" + realValue[0];
		str = str + "|" + realValue[1];
		str = str + "|" + realValue[2];
		str = "<" + str.substring(1) + ">";
		return str;
	}
	
	public void valueOf(String str) {
		str = str.substring(1, str.length()-1);
		String vStr[] = str.split("\\|");
		for (int i=0; i<=4; i++) {
			discValue[i] = Integer.valueOf(vStr[i]);
		}
		for (int i=0; i<=10; i++) {
			intValue[i] = Integer.valueOf(vStr[i+5]);
		}
		realValue[0] = Double.valueOf(vStr[16]);
		realValue[0] = Double.valueOf(vStr[17]);
		realValue[0] = Double.valueOf(vStr[18]);
	}
}
