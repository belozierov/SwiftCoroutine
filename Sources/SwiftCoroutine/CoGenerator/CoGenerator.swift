//
//  CoGenerator.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 30.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation

class CoGenerator<Output>: CoFuture<Output> {
    
    typealias Iterator = ((OutputResult) -> Void) -> Void
    typealias StackSize = Coroutine.StackSize
    
    let coroutine: Coroutine
    let iterator: Iterator
    
    init(dispatcher: Dispatcher = .sync,
         stackSize: StackSize = .recommended,
         iterator: @escaping Iterator) {
        coroutine = Coroutine(dispatcher: dispatcher, stackSize: stackSize)
        self.iterator = iterator
    }
    
    var requestQueue = [(OutputResult) -> Void]()
    
    
    
}
