package Mushroom;

public class MushroomValue {
	public int[] value;
	
	public MushroomValue() {
		value = new int[22];
		for (int i=0; i<=21; i++) {
			value[i]=0;
		}
	}
	
	public MushroomValue(int[] v) {
		value = new int[22];
		for (int i=0; i<=21; i++) {
			value[i]=v[i];
		}
	}
	
	public String toString() {
		String str = "";
		for (int i=0; i<=21; i++) {
			str = str +"|" + value[i];
		}
		str = "<" + str.substring(1) + ">";
		return str;
	}
	
	public void valueOf(String str) {
		str = str.substring(1, str.length()-1);
		String vStr[] = str.split("\\|");
		for (int i=0; i<=21; i++) {
			value[i] = Integer.valueOf(vStr[i]);
		}
	}
}
