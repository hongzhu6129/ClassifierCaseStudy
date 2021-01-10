package Mushroom;

public final class MushroomFeatures {
	public final static int cap_shape = 0; 	/* bell=b, conical=c, flat=f, knobbed=k, sunken=s, convex=x	*/
	public final static int cap_surface =1; /* fibrous=f, grooves=g, smooth=s, scaly=y*/
	public final static int cap_color =2;	/* buff=b, cinnamon=c,  red=e, gray=g, brown=n, pink=p, green=r, purple=u, white=w, yellow=y */
	public final static int bruises =3;		/* no=f, bruises=t */
	public final static int odor =4;		/* almond=a, creosote=c, foul=f, anise=l, musty=m, none=n, pungent=p, spicy=s, fishy=y */
	public final static int gill_attachment =5;	/* attached=a, descending=d, free=f, notched=n */
	public final static int gill_spacing =6;	/* close=c, distant=d, crowded=w */
	public final static int gill_size = 7;	/* broad=b, narrow=n */
	public final static int gill_color = 8;	/* buff=b,  red=e, gray=g, chocolate=h, black=k, brown=n, orange=o, pink=p, green=r, purple=u, white=w, yellow=y */
	public final static int stalk_shape =9;	/* enlarging=e, tapering=t	*/
	public final static int stalk_root =10;	/* missing=?, bulbous=b, club=c, equal=e, rooted=r, cup=u, rhizomorphs=z */			
	public final static int stalk_surface_above_ring =10; /* fibrous=f, silky=k, smooth=s, scaly=y */
	public final static int stalk_surface_below_ring =11; /* fibrous=f, silky=k, smooth=s, scaly=y */
	public final static int stalk_color_above_ring =12; /* buff=b, cinnamon=c, red=e, gray=g, brown=n, orange=o, pink=p, white=w, yellow=y */
	public final static int stalk_color_below_ring =13; /* buff=b, cinnamon=c, red=e, gray=g, brown=n, orange=o, pink=p, white=w, yellow=y */
	public final static int veil_type =14; 	/* partial=p, universal=u */
	public final static int veil_color =15; 	/* brown=n, orange=o, white=w, yellow=y */
	public final static int ring_number =16;	/* none=n, one=o, two=t */
	public final static int ring_type =17; 	/* cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z */
	public final static int spore_print_color =19; 	/* buff=b, chocolate=h, black=k, brown=n, orange=o, green=r, purple=u, white=w, yellow=y */
	public final static int population =20; 	/* abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y */
	public final static int habitat =21; 	/* woods=d, grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w */

	public final static int[] size = {6, 4, 10, 2, 9, 4, 3, 2, 12, 2, 6, 4, 4, 9, 9, 2, 4, 3, 3, 8, 9, 6, 7}; 
	
	/*
	public MushroomFeatures() {
		cap_shape = 0; 	
		cap_surface = 0; 
		cap_color = 0;
		bruises = 0;
		odor = 0;
		gill_attachment = 0;
		gill_spacing = 0;
		gill_size = 0;	
		gill_color = 0;
		stalk_shape = 0;
		stalk_root = 0;	
		stalk_surface_above_ring = 0; 
		stalk_surface_below_ring = 0; 
		stalk_color_above_ring = 0; 
		stalk_color_below_ring = 0; 
		veil_type = 0; 	
		veil_color = 0; 	
		ring_number = 0;	
		ring_type = 0; 
		spore_print_color = 0; 
		population = 0; 	
		habitat = 0; 	
	}
	
	public MushroomFeatures(
			int p_cap_shape, 	
			int p_cap_surface,
			int p_cap_color,
			int p_bruises,
			int p_odor,
			int p_gill_attachment,
			int p_gill_spacing,
			int p_gill_size,
			int p_gill_color,
			int p_stalk_shape,
			int p_stalk_root,
			int p_stalk_surface_above_ring,
			int p_stalk_surface_below_ring,
			int p_stalk_color_above_ring,
			int p_stalk_color_below_ring,
			int p_veil_type,
			int p_veil_color,
			int p_ring_number,
			int p_ring_type,
			int p_spore_print_color,
			int p_population,
			int p_habitat) {
		cap_shape = p_cap_shape; 	
		cap_surface = p_cap_surface; 
		cap_color = p_cap_color;	
		bruises = p_bruises;
		odor = p_odor;
		gill_attachment = p_gill_attachment;
		gill_spacing = p_gill_spacing;
		gill_size = p_gill_size;	
		gill_color = p_gill_color;
		stalk_shape = p_stalk_shape;
		stalk_root = p_stalk_root;	
		stalk_surface_above_ring = p_stalk_surface_above_ring; 
		stalk_surface_below_ring = p_stalk_surface_below_ring; 
		stalk_color_above_ring = p_stalk_color_above_ring; 
		stalk_color_below_ring = p_stalk_color_below_ring; 
		veil_type = p_veil_type; 
		veil_color = p_veil_color; 
		ring_number = p_ring_number;
		ring_type = p_ring_type; 
		spore_print_color = p_spore_print_color; 
		population = p_population;
		habitat = p_habitat; 
	}
	
	public String toString() {
		String str = "<" 
				+ cap_shape + "|"
				+ cap_surface + "|"
				+ cap_color + "|"
				+ bruises + "|"
				+ odor + "|"
				+ gill_attachment + "|"
				+ gill_spacing + "|"
				+ gill_size + "|"
				+ gill_color + "|"
				+ stalk_shape +"|"
				+ stalk_root + "|"
				+ stalk_surface_above_ring + "|"
				+ stalk_surface_below_ring + "|"
				+ stalk_color_above_ring + "|"
				+ stalk_color_below_ring + "|"
				+ veil_type + "|"
				+ veil_color + "|"
				+ ring_number + "|"
				+ ring_type + "|"
				+ spore_print_color + "|"
				+ population + "|"
				+ habitat + ">";
		return str;
	} 
	
	public void valueOf(String str) {
		str = str.substring(1, str.length()-1);
		String vStr[] = str.split("\\|");
		cap_shape = Integer.valueOf(vStr[0]);
		cap_surface = Integer.valueOf(vStr[1]);
		cap_color = Integer.valueOf(vStr[2]);
		bruises = Integer.valueOf(vStr[3]);
		odor = Integer.valueOf(vStr[4]);
		gill_attachment = Integer.valueOf(vStr[5]);
		gill_spacing = Integer.valueOf(vStr[6]);
		gill_size = Integer.valueOf(vStr[7]);
		gill_color = Integer.valueOf(vStr[8]);
		stalk_shape = Integer.valueOf(vStr[9]);
		stalk_root = Integer.valueOf(vStr[10]);
		stalk_surface_above_ring = Integer.valueOf(vStr[11]); 
		stalk_surface_below_ring = Integer.valueOf(vStr[12]);
		stalk_color_above_ring = Integer.valueOf(vStr[13]);
		stalk_color_below_ring = Integer.valueOf(vStr[14]);
		veil_type = Integer.valueOf(vStr[15]);
		veil_color = Integer.valueOf(vStr[16]);
		ring_number = Integer.valueOf(vStr[17]);
		ring_type = Integer.valueOf(vStr[18]);
		spore_print_color = Integer.valueOf(vStr[19]);
		population = Integer.valueOf(vStr[20]);
		habitat = Integer.valueOf(vStr[21]);
	}
/*
/*
	cap_shape;
	cap_surface;
	cap_color;
	bruises;
	odor;
	gill_attachment;
	gill_spacing;
	gill_size;
	gill_color;
	stalk_shape;
	stalk_root;
	stalk_surface_above_ring;
	stalk_surface_below_ring;
	stalk_color_above_ring;
	stalk_color_below_ring;
	veil_type;
	veil_color;
	ring_number;
	ring_type;
	spore_print_color;
	population;
	habitat;
*/
}
