#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 
#define LOGLEN 20


struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 

pthread_mutex_t mutex;
int state;
int log_pos[9];
char map[ROW+10][COLUMN] ; 

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

void printMap(void){
	system("clear");
	for(int i = 0; i <= ROW; ++i)	
		puts( map[i] );
}

void *logs_move( void *t ){

	/*  Move the logs  */
	int count = 0;
	while(state == 0 && count < 10){
		for(int i = 1; i < ROW ; i++){
			// odd row go right
			if(i % 2 == 1){
				log_pos[i] ++;
				if( log_pos[i] > COLUMN-1)
					log_pos[i] = 0;
			}
			// even row go left
			else{
				log_pos[i]--;
				if(log_pos[i] < 0)
					log_pos[i] = COLUMN -1;
			}
		}

		// translate the logpos array to the map array
		for(int i = 1; i < ROW; i++){
			// did not exceed the edge
			if(log_pos[i] + LOGLEN < COLUMN -1){
				for(int j = 0; j < COLUMN-1; j++){
					if(j>=log_pos[i] && j <= log_pos[i]+LOGLEN){
						map[i][j] = '=';
					}
					else	
						map[i][j] = ' ';
				}
			}
			// exceed the edge
			else{
				for(int j = 0; j < COLUMN-1; j++){
					if(j<log_pos[i]+LOGLEN - COLUMN || j > log_pos[i]){
						map[i][j] = '=';
					}
					else
						map[i][j] = ' ';
				}
			}
		}
		printMap();
		sleep(1);
		count ++;
	}


	/*  Check keyboard hits, to change frog's position or quit the game. */

	
	/*  Check game's status  */


	/*  Print the map on the screen  */

	return NULL;
}

void *frog_control(void* arg){
	return NULL;
}


void init_logs(void){
	time_t t;
	// initialize the random generator
	srand((unsigned) time(&t));
	for(int i = 0; i<9 ; ++i){
		log_pos[i] = rand()%(COLUMN);
	}
}

int main( int argc, char *argv[] ){

	// Initialize the state of the game 
	state = 0;
	
	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	// Initialize the logs
	init_logs();

	//Print the map into screen
	printMap();

	/*  Create pthreads for wood move and frog control.  */
	int rc1, rc2;

	pthread_t log_t, frog_t;
	rc1 = pthread_create(&log_t, NULL, logs_move, NULL);
	rc2 = pthread_create(&frog_t, NULL, frog_control, NULL);
	/*  Display the output for user: win, lose or quit.  */
	if(rc1){
		printf("ERROR: return code from pthread_create() is %d", rc1);
		exit(1);
	}
	if(rc2){
		printf("ERROR: return code from pthread_create() is %d", rc2);
		exit(1);
	}

	/* Wait for game to end */
	pthread_join(log_t, NULL);
	pthread_join(frog_t, NULL);

	pthread_mutex_destroy(&mutex);
	pthread_exit(NULL);

	return 0;

}
