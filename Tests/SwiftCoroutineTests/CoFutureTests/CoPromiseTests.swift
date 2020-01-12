//
//  CoPromiseTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 10.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoPromiseTests: CoFutureTests {
    
    override var future: CoPromise<Bool> {
        CoPromise()
    }
    
    func testSend() {
        let future = self.future
        XCTAssertNil(future.result)
        future.send(true)
        future.send(false)
        future.cancel()
        XCTAssertEqual(future.result, true)
        XCTAssertEqual(future.resultStorage, true)
    }
    
}
