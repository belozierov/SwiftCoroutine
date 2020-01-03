//
//  CoFuture+Error.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

extension CoFuture {
    
    public enum FutureError: Error {
        case cancelled, awaitCalledOutsideCoroutine, timeout
    }
    
}
