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

	while(state == 0){
		pthread_mutex_lock(&mutex);
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
		// frog shift with the log
		if(frog.x >0 && frog.x <10){
			if(frog.x % 2 == 1)
				frog.y ++;
			else
				frog.y --;
		}
		map[frog.x][frog.y] = '0';
		/*  Print the map on the screen  */
		printMap();

		/*  Check game's status  */

		// left or right out
		if(frog.y > COLUMN-1 || frog.y < 0)
			state = -1;
		// if frog on bank
		else if(frog.x == 0)
			state = 1;
		else if(frog.x == 10)
			state = 0;		
		// if frog on entire log 
		else if(log_pos[frog.x] + LOGLEN < COLUMN -1){
			if(frog.y>=log_pos[frog.x] && frog.y <= log_pos[frog.x]+LOGLEN)
				state = 0;
			else
				state = -1;
		}
		// if frog on half log
		else{
			if(frog.y<log_pos[frog.x]+LOGLEN - COLUMN || frog.y > log_pos[frog.x])
				state = 0;
			else
				state = -1;
		}

		if(state == -1)
			printf("YOU LOSE...");
		if(state == 1)
			printf("YOU WIN...");
		pthread_mutex_unlock(&mutex);
		sleep(1);
	}
	state = -1;
	return NULL;
}

void *frog_control(void* arg){

	/*  Check keyboard hits, to change frog's position or quit the game. */
	while(state == 0){
		int pre_x = frog.x;
		int pre_y = frog.y;
		if(kbhit()){
			printf("previous frog:%d, %d", frog.x, frog.y);

			pthread_mutex_lock(&mutex);
			char kb;
			kb = getchar();
			if(kb == 'q' || kb == 'Q'){
				state = 2;
				printf("QUIT GAME...");
			}
			if(kb == 'w' || kb == 'W'){
				frog.x --;
			}
			if(kb == 's' || kb == 'S'){
				frog.x ++;
			}
			if(kb == 'a' || kb == 'A'){
				frog.y --;
			}
			if(kb == 'd' || kb == 'D'){
				frog.y ++;
			}
			printf("key board catch: %d", kb);
			printf("now frog:%d, %d", frog.x, frog.y);

			// update the map
			map[frog.x][frog.y] = '0';
			// if the frog is on the bank
			if(pre_x == 10 || pre_x == 0)
				map[pre_x][pre_y] = '|';

			// the frog is on a log
			else
				map[pre_x][pre_y] = '=';
			pthread_mutex_unlock(&mutex);
		}
	}
	
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
	pthread_t log_t, frog_t;

	pthread_mutex_init(&mutex, NULL);

	pthread_create(&log_t, NULL, logs_move, NULL);
	pthread_create(&frog_t, NULL, frog_control, NULL);
	/*  Display the output for user: win, lose or quit.  */

	/* Wait for game to end */
	pthread_join(log_t, NULL);
	pthread_join(frog_t, NULL);

	pthread_mutex_destroy(&mutex);
	pthread_exit(NULL);

	return 0;

}
